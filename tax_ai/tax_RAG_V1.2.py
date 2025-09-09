import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from transformers import TextStreamer, TextIteratorStreamer
import datetime
import re
from langchain_core.documents import Document
import gradio as gr
import jieba
import pandas as pd
import itertools
import time

Embedding_Model = './models/bge-small-zh-v1.5'
#LLM_Model = './models/Qwen1.5-14B-Chat-GPTQ-Int4'
LLM_Model = './models/DeepSeek-R1-Distill-Qwen-14B'
device = "cuda" # the device to load the model onto
CUDA_Device = 'cuda:0'

file_path = './data/Tax-File'
store_path = './data/tax.faiss'
filter_path = './data/cn_stopwords.txt'
test_file_path = './data/test.xlsx'
year_range = ['2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
tax_range = ['增值税','消费税','企业所得税','个人所得税','进出口税收','资源税','印花税','烟叶税','消费税','土地增值税','契税','环境保护税','耕地占用税','房产税','城镇土地使用税','城市维护建设税','车辆购置税','车船税','船舶吨税']

metadata_filter = True #是否在向量检索前使用元数据进行chunks过滤
chunk_mode = 'soft' #数据分块的切分模式：[hard, soft]，hard模式为基于chunk size进行划分，soft模式为基于数据组织特点和文本内容分析进行划分
test_mode = 'single' #测试模式：[single, batch]，通过demo进行单条测试，还是读取excel文件进行批量测试

use_deepseek_local = True #if to use deepseek local llm when query is not tax-related

#分词
def cut_word(Test):
    # jieba 默认启用了HMM（隐马尔科夫模型）进行中文分词
    seg_list = jieba.cut(Test,cut_all=True)  # 分词
    word = out_stopword(seg_list)
    #列出关键字
    # print("原内容：")
    # print(Test)
    # print("分词过滤后：")
    # print(word)
    return word

#去除停用词
def out_stopword(seg):
    wordlist = []

    #获取停用词表
    stop = open(filter_path, 'r+', encoding='utf-8')
    #用‘\n’去分隔读取，返回一个一维数组
    stopword = stop.read().split("\n")
    #遍历分词表
    for key in seg:
        #去除停用词，去除单字，去除重复词
        if not(key.strip() in stopword) and (len(key.strip()) > 1) and not(key.strip() in wordlist) :
            wordlist.append(key)
            
    #停用词去除END
    stop.close()
    return wordlist


def get_neighbor_range(keyword, context, n, direction):
  '''
  keyword: 定位词
  context: 上下文
  n: 与定位词相邻多少个字符
  direction：相邻上文left，或相邻下文right
  return 相邻片段
  '''
  span = ''
  if context.find(keyword) < 0:
    raise Exception("上下文中没有找到指定的定位词！")
  loc = context.find(keyword)
  if direction == 'left':
    if n > loc:
      span = context[:loc]
    else:
      span = context[loc-n:loc]
  else:
    if n > len(context) - (loc + len(keyword)):
      span = context[loc + len(keyword):]
    else:
      span = context[loc + len(keyword):loc + len(keyword) + n]

  return span




def chunk_soft(content):
  chunk_list = []

  if content:
    #提取公告日期
    file_date = ""
    if "【打印】" in content:
      content = content.split("【打印】")[0]
    if len(content) >=20:
      if re.search(r'\d{4}年\d{1,2}月\d{1,2}日', content[-20:]):
        file_date = re.search(r'\d{4}年\d{1,2}月\d{1,2}日', content[-20:]).group(0)

    #清洗文本，基于开头和结尾关键词进行截断，去掉无效的开头和结尾，保留核心条款内容
    start_ky = ["通知如下","公告如下","明确如下"]
    end_ky = ["特此公告","特此通知","特此说明","附件："]

    for keyword in start_ky:
      if keyword in content:
        content = content.split(keyword)[1]
        break
    for keyword in end_ky:
      if keyword in content:
        if keyword != "附件：" and len(content.split(keyword)[1]) > 200: #不进行结尾去除
          break
        content = content.split(keyword)[0]
        break
    if len(content) >= 20:
      if "\n财政部" in content[-20:]:
        content = content.split("\n财政部")[0]
      elif "\n国务院" in content[-20:]:
        content = content.split("\n国务院")[0]

    #按照条款级别，进行细粒度划分
    splitter = RecursiveCharacterTextSplitter(
  chunk_size=100, chunk_overlap=0, separators=["\n\n", r"[一二三四五六七八九十百千]+、",r"第[一二三四五六七八九十百千]+条",r"（[一二三四五六七八九十百千]+）","\n"]
      )
    clauses = splitter.split_text(content)

    #检测上下文指代和时效性信息
    clause_list = []

    for clause in clauses:
      tmp_dict = {}
      tmp_dict["ori_cont"] = clause

      #上下文指代
      if "上述" in clause:
        tmp_dict["is_refer"] = True
      else:
        tmp_dict["is_refer"] = False

      #标准化时效性信息
      new_cont = clause

      date_ky = ["发布之日","公布之日","印发之日","发文之日"]
      act_ky = ["执行","施行","实施","延长"]
      if file_date:
        for keyword in date_ky:
          if keyword in new_cont and any([key in get_neighbor_range(keyword, new_cont, 50, 'right') for key in act_ky+["废止"]]):
            new_cont = new_cont.replace(keyword,file_date)
            break
        if "同时废止" in new_cont:
          new_cont = new_cont.replace("同时",file_date)
        if "废止。" in new_cont and not re.search(r'\d{4}年\d{1,2}月\d{1,2}日', get_neighbor_range("废止。", new_cont, 50, 'left')):
          new_cont = new_cont.split("废止。")[0] + file_date + "废止。" + new_cont.split("废止。")[1]

      chunk_list.append(new_cont)

      #将条款划分为句子片段
      spans = re.split('。|，',new_cont)

      #判断是否附加时效性信息，提炼出需要附加的时效性信息（case1）
      add_case1 = ""
      file_ky = ["本公告","本通知","本办法","本说明","上述"]
      for keyword in file_ky:
        if keyword in new_cont and any([key in get_neighbor_range(keyword, new_cont, 50, 'right') for key in act_ky]):
          for index, span in enumerate(spans):
            if keyword in span and "废止" not in span: #找到关键词所在的句子片段
              #按照长度阈值整合关键词所在的句子片段和下一句子片段
              if index != len(spans)-1 and new_cont[new_cont.find(span)+1] != '。':
                if len(span) + len(spans[index+1]) < 50:
                  span = span + '，' + spans[index+1]

              #判断span是否超出长度阈值，是否包含日期，日期在关键词后
              if len(span) < 50 and re.search(r'\d{4}年\d{1,2}月\d{1,2}日',span):
                date = [match.start() for match in re.finditer('\d{4}年\d{1,2}月\d{1,2}日', span)]
                ky = [match.start() for match in re.finditer(keyword, span)]
                if ky[0] < date[0]:
                  for pattern in [r"[一二三四五六七八九十百千]+、",r"第[一二三四五六七八九十百千]+条",r"（[一二三四五六七八九十百千]+）"]:
                    span = re.sub(pattern, "", span)
                  add_case1 = span

      tmp_dict["add_case1"] = add_case1

      #判断是否附加时效性信息，提炼出需要附加的时效性信息（case2）
      add_case2 = ""
      for span in spans:
        if any([key in span for key in ["下列文件或文件条款","下列文件或者文件条款"]]):
          if "下列文件或文件条款" in span:
            add_case2 = span.replace("下列文件或文件条款","该文件条款")
          if "下列文件或者文件条款" in span:
            add_case2 = span.replace("下列文件或者文件条款","该文件条款")
          break

      tmp_dict["add_case2"] = add_case2

      clause_list.append(tmp_dict)

    #基于指代关键词进行合并
    for index, tmp_dict in enumerate(clause_list):
      if tmp_dict["is_refer"] and index > 0: #将本条合并到上一条
        if chunk_list[index-1] and len(chunk_list[index-1] + chunk_list[index]) < 256:
          chunk_list[index-1] = chunk_list[index-1] + chunk_list[index]
          clause_list[index-1]["add_case1"] = max(clause_list[index-1]["add_case1"], clause_list[index]["add_case1"])
          clause_list[index-1]["add_case2"] = max(clause_list[index-1]["add_case2"], clause_list[index]["add_case2"])
          chunk_list[index] = "" #本条置为空
          clause_list[index] = None

        else:
          if index > 1 and chunk_list[index-2] and len(chunk_list[index-2] + chunk_list[index]) < 256:
            chunk_list[index-2] = chunk_list[index-2] + chunk_list[index]
            clause_list[index-2]["add_case1"] = max(clause_list[index-2]["add_case1"], clause_list[index]["add_case1"])
            clause_list[index-2]["add_case2"] = max(clause_list[index-2]["add_case2"], clause_list[index]["add_case2"])
            chunk_list[index] = "" #本条置为空
            clause_list[index] = None

    #附加时效性信息
    add_case2_index = 9999999
    for index, tmp_dict in enumerate(clause_list):
      if tmp_dict and tmp_dict["add_case2"] != "":
        #将该条款后的所有条款附加该时效性信息
        add_case2_index = index
        chunk_list = [ new_cont + tmp_dict["add_case2"] if index < loc and new_cont else new_cont for loc, new_cont in enumerate(chunk_list)]

    for index, tmp_dict in enumerate(clause_list):
      if tmp_dict and tmp_dict["add_case1"] != "":
        #将其他每个条款附加该时效性信息
        chunk_list = [ new_cont + tmp_dict["add_case1"] if index != loc and loc < add_case2_index and new_cont else new_cont for loc, new_cont in enumerate(chunk_list)]

  else:
    raise Exception("待chunk的内容为空！")

  return list(filter(lambda x: x != '', chunk_list))




# Split the text into docs
def split_text(txt, chunk_size=256, overlap=32):
    if not txt:
        raise Exception("入库文件内容为空！")

    docs = []
    for doc in txt:
        print("Start to split text...")
        print(doc.metadata['source'])

        file_number = doc.page_content.split('\n')[0]
        if chunk_mode == 'hard':
          splitter = RecursiveCharacterTextSplitter(
              chunk_size=chunk_size, chunk_overlap=overlap
          )
          md_docs = splitter.split_text(doc.page_content)
        elif chunk_mode == 'soft':
          md_docs = chunk_soft(doc.page_content)
        else:
          raise Exception("仅支持hard或soft的数据切分模式！")

        for md_doc in md_docs: #用文档名称的分词结果作为keywords
            doc.metadata.update({"nr": file_number})
            doc.metadata.update( {"keywords": cut_word(doc.metadata['source'].split('/')[-1].split('.')[0])})
            doc.metadata.update({"year": doc.metadata['source'].split('/')[-1].split('.')[0].split('_')[-1]})
            doc.metadata.update({"tax": doc.metadata['source'].split('/')[-2]})
            docs.append(Document(page_content=md_doc, metadata=doc.metadata))

    return docs



def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=Embedding_Model,
        model_kwargs={'device': CUDA_Device}
    )
    return embeddings


# Save docs to vector store with embeddings
def create_vector_store(docs, embeddings, store_path):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store


# Load vector store from file
def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None


def load_file(file_path):
    loader = DirectoryLoader(file_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    docs = loader.load()
    print(len(docs))
    return docs

def load_or_create_vector_store(store_path, file_path):
    embeddings = create_embeddings()
    vector_store = load_vector_store(store_path, embeddings)
    if not vector_store:
        # Not found, build the vector store
        txt = load_file(file_path)
        docs = split_text(txt)
        vector_store = create_vector_store(docs, embeddings, store_path)

    return vector_store


# Query the context from vector store
def query_vector_store(vector_store, query, year_begin, year_end, tax, k=3, relevance_threshold=0.7):
    cut_list = cut_word(query)
    if metadata_filter:
      if year_begin and tax:        
        similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k, 
    filter=lambda d: list(set(d["keywords"]) & set(cut_list)) and (d["year"]>= year_begin and d["year"]<= year_end) and d["tax"]==tax)
      elif year_begin and not tax:
        similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k, 
    filter=lambda d: list(set(d["keywords"]) & set(cut_list)) and (d["year"]>= year_begin and d["year"]<= year_end))
      elif not year_begin and tax:
        similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k, 
    filter=lambda d: list(set(d["keywords"]) & set(cut_list)) and d["tax"]==tax)
      else:
        similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k, 
    filter=lambda d: list(set(d["keywords"]) & set(cut_list)))        
    else:
      similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)

    print('similar_docs:')
    print(similar_docs)

    need_tax = False
    context = []
    metadata = []
    scores = []
    # 基于税务关键词没有命中知识块时，需要进行query引导，增添税种类别
    if not similar_docs and not tax:
      need_tax = True
    else:
      related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
      if not related_docs and not tax:
        need_tax = True
      else:
        context = [doc[0].page_content for doc in related_docs]
        metadata = [doc[0].metadata for doc in related_docs]
        scores = [str(doc[1]) for doc in related_docs]

    return context, metadata, scores, need_tax

def query_rewrite_and_check(query):
  year_begin = ''
  year_end = ''
  tax = ''
  is_within = True

  # 年份标准化
  if re.search(r'\d{4}(?:年)?(?:前|之前)', query):
    tmp = re.search(r'(\d{4})(?:年)?(?:前|之前)', query).group(1)
    year_begin = year_range[0]
    year_end = str(int(tmp) - 1)
    if year_end < year_range[0]:
      is_within = False

  elif re.search(r'\d{4}(?:年)?(?:后|之后)', query):
    tmp = re.search(r'(\d{4})(?:年)?(?:后|之后)', query).group(1)
    year_begin = str(int(tmp) + 1)
    year_end = year_range[-1]
    if year_begin > year_range[-1]:
      is_within = False

  elif re.search(r'\d{4}[年]?(?:到|至)\d{4}[年]?(?:之间|间)', query):
    year_begin = re.search(r'(\d{4})[年]?(?:到|至)(\d{4})[年]?(?:之间|间)', query).group(1)
    year_end = re.search(r'(\d{4})[年]?(?:到|至)(\d{4})[年]?(?:之间|间)', query).group(2)
    
    if year_begin > year_range[-1] or year_end < year_range[0]:
      is_within = False

  elif re.search(r'(\d{4})(?:年)', query):
    tmp = re.search(r'(\d{4})(?:年)', query).group(1)
    year_begin = tmp
    year_end = tmp
    if year_begin > year_range[-1] or year_end < year_range[0]:
      is_within = False
  elif any([keyword in query for keyword in ['最新','最近','近期','近来']]): 
    tmp = str(datetime.datetime.now().year)
    year_begin = tmp
    year_end = tmp
    if year_begin > year_range[-1]:
      is_within = False
  

  # 税种
  for each in tax_range:
    if each in query:
      tax = each
      break

  return is_within, year_begin, year_end, tax 


# Load model
def load_llm(model_path):

  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
    )
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  return model, tokenizer

def ask(model, tokenizer, prompt, max_tokens=1024):

  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
  model_inputs = tokenizer([text], return_tensors="pt").to(device)
  input_ids = tokenizer.encode(text,return_tensors='pt')
  attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=device)

  generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1024,
    pad_token_id=tokenizer.eos_token_id  
    )
  generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  return response


def main_chain(question):
    query_check = ''
    query_retrieve = ''
    query_ans = ''

    # query检查
    prompt = f'你是专业的税务顾问，请判断下面问题是否和税务相关，相关请回答“是”，不相关请回答“否”，不允许其它回答，不允许回答为空。问题: \n{question}\n'
    ans = ask(model, tokenizer, prompt)
    print("ans of query check stage:")
    print(ans)
    query_check = ans

    # query改写和快速检查（年份，税种）
    is_within, year_begin, year_end, tax = query_rewrite_and_check(question)
    if is_within:
      # query检索和生成答案
      context, metadata, scores, need_tax = query_vector_store(vector_store, question, year_begin, year_end, tax, 20, 0.6)
      print('len of context:')
      print(len(context))
      query_retrieve = '$'.join([str(a) + ',context:' + b for a,b in zip(metadata, context)])
        
      if len(context) != 0:
        prompt = f'你是专业的税务顾问，请你结合以下材料回答问题，答案里不允许添加材料范围外的内容，答案里要包含材料中提及的所有时间实体，日期实体和数字实体，材料：\n{context}\n问题: \n{question}\n'
        # eg: 增值税，财政部 税务总局公告2023年第54号，《关于延续实施小额贷款公司有关税收优惠政策》：
        #    一、对经省级地方金融监督管理部门批准成立的小额贷款公司取得的农户小额贷款利息。。。

        ans = ask(model, tokenizer, prompt)
        print("ans of query index and answer stage:")
        print(ans)
        query_ans = ans

    return query_check + '#' + query_retrieve + '#' + query_ans


def batch_test():

    df = pd.read_excel(test_file_path)
    print("the num of test queries from test file:")
    print(len(df.index.values)) # records num

    df['ori_result'] = df.apply(lambda x: main_chain(x['问题']), axis=1)
    df[['相关性回答','检索内容TOP','生成答案']] = df.ori_result.str.split('#', expand = True)
    df.drop(axis = 1, columns = 'ori_result', inplace = True)

    writer = pd.ExcelWriter("batch_test_result.xlsx")
    df.to_excel(writer)
    writer._save()
    writer.close()



def chat(message, history, progress=gr.Progress()):

    question = message
    retrieve_ans = ""
    generate_ans = ""

    # query检查
    prompt = f'你是专业的税务顾问，请判断下面问题是否和税务相关，相关请回答“是”，不相关请回答“否”，不允许其它回答，不允许回答为空。问题: \n{question}\n'
    topic_check_ans = ask(model, tokenizer, prompt)
    if 'DeepSeek' in LLM_Model:
        topic_check_ans_final = topic_check_ans.split('</think>')[-1].replace('\n','')
    else:
        topic_check_ans_final = topic_check_ans
    print("ans of query check stage:")
    print(topic_check_ans_final)

    if topic_check_ans_final in ['否','否。'] and not use_deepseek_local:
        with open('log/call_log.txt','a+') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) + ' ' + question + '&' + topic_check_ans_final + '&' + retrieve_ans + '&' + generate_ans + '\n')
        output = "抱歉，我是税务AI助手，请提问和税务有关的问题。"
        
        final_output = ""
        for new_text in output:
          final_output += new_text
          yield final_output
    else: 
      need_tax = False

      if topic_check_ans_final in ['否','否。'] and use_deepseek_local:
          prompt = f'你是专业助手，请回答以下问题: \n{question}\n'
          source = "您的提问和税务主题无关，本次回答不依据税务知识库，仅依据大模型的相关储备知识。"
          retrieve_ans = ""
      else:
          # query改写和快速检查（年份，税种）
          is_within, year_begin, year_end, tax = query_rewrite_and_check(question)
          if not is_within:
              prompt = f'你是专业的税务顾问，请回答以下问题: \n{question}\n'
              source = "您的提问超出知识库范围，目前支持的范围如下：年份（"+year_range[0]+"至"+year_range[-1]+"），税种（"+'，'.join(tax_range)+"），本次回答不依据知识库。"
              retrieve_ans = ""

          else:
            # query检索和生成答案
            context, metadata, scores, need_tax = query_vector_store(vector_store, question, year_begin, year_end, tax, 20, 0.6)
            print('len of context:')
            print(len(context))

            if need_tax:
              output = "抱歉，请您在问题中明确税种信息，然后重新提问。目前支持的税种如下："+'，'.join(tax_range)+"。"
              with open('log/call_log.txt','a+') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) + ' ' + question + '&' + topic_check_ans_final + '&' + retrieve_ans + '&' + output+'\n')

              final_output = ""
              for new_text in output:
                final_output += new_text
                yield final_output
            else:
              if len(context) == 0:
                  # 没有检索到质量高的相关知识内容，直接回答问题
                  prompt = f'你是专业的税务顾问，请回答以下问题: \n{question}\n'
                  source = "未能从知识库中检索到相关性高的内容，本次回答不依据知识库。"
              else:
                  context_all = '\n'.join(context)
                  prompt = f'你是专业的税务顾问，请你结合以下材料回答问题，答案里不允许添加材料范围外的内容，答案里要包含材料中提及的所有时间实体，日期实体和数字实体，材料：\n{context}\n问题: \n{question}\n'
                  # eg: 增值税，财政部 税务总局公告2023年第54号，《关于延续实施小额贷款公司有关税收优惠政策》：
                  #    一、对经省级地方金融监督管理部门批准成立的小额贷款公司取得的农户小额贷款利息。。。

                  source_dict = {}
                  #对来源进行整合，便于展示
                  for content,each in zip(context,metadata):
                    key = each["source"].split('/')[-2] + "，" + each["nr"] + "，" + "《" + each["source"].split('/')[-1].split('_')[0] +"》"
                    if key not in source_dict.keys():
                      source_dict[key] = [content.strip(" ：\n").replace("\n","")]
                    else:
                      source_dict[key].append(content.strip(" ：\n").replace("\n",""))

                  source = "\n".join(["**（"+str(index+1)+"）" + key + "**\n" + "\n".join(["_“" + x + "”_" for x in source_dict[key]]) for index,key in enumerate(source_dict.keys())])

              retrieve_ans = '$'.join([str(a) + ',context:' + b + ',score:' + c for a,b,c in zip(metadata, context, scores)])

      if not need_tax:

        # 生成答案的流式输出
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
          )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # 使用流式传输模式（更加流畅和动态的交互体验）
        # 自定义、可迭代的流式输出
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        attention_mask = torch.ones(model_inputs.input_ids.shape,dtype=torch.long,device=device)

        # Use Thread to run generation in background
        # Otherwise, the process is blocked until generation is complete
        # and no streaming effect can be observed.
        from threading import Thread
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024, attention_mask=attention_mask)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 循环推理streamer，Gradio会监控generated_text变量在页面展示
        generated_text = ''
        for new_text in streamer:
          generated_text += new_text
          yield generated_text

        if 'DeepSeek' in LLM_Model:
            generate_ans = generated_text.split('</think>')[-1]
            how_to_think = generated_text.split('</think>')[0]
        else:
            generate_ans = generated_text
            how_to_think = ''

        if 'DeepSeek' in LLM_Model:
            ans_and_how_to_think = '''<font size=5>答案  *****************************************************************************************************</font>\n''' + generate_ans + "\n\n\n" + '''<font size=5>思考过程  *****************************************************************************************************</font>\n''' + how_to_think
        else:
            ans_and_how_to_think = '''<font size=5>答案  *****************************************************************************************************</font>\n''' + generate_ans

        output = ans_and_how_to_think + "\n\n\n" + '''<font size=5>来源  *****************************************************************************************************</font>\n''' + source
        source_output = ""
        for new_text in output:
          source_output += new_text
          yield source_output

        
        with open('log/call_log.txt','a+') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()) + ' ' + question + '&' + topic_check_ans_final + '&' + retrieve_ans + '&' + generate_ans+'\n')


def vote(data: gr.LikeData):
    if data.liked:
      with open('log/call_log.txt','a+') as f:
        f.write("You upvoted this response.\n")
        # print("You upvoted this response: " + data.value["value"])
    else:
      with open('log/call_log.txt','a+') as f:
        f.write("You downvoted this response.\n")
        # print("You downvoted this response: " + data.value["value"])



if __name__ == '__main__':

    vector_store = load_or_create_vector_store(store_path, file_path)

    model, tokenizer = load_llm(LLM_Model)


    CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 3; overflow: auto;}
    """
    if test_mode == 'single':
      with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
       # with gr.Column(elem_classes="title"):
        #  gr.Markdown(" # <center>税务AI小助手</center>")

        #chatbot = gr.Chatbot(
         #   elem_id="chatbot",
          #  bubble_full_width=False,
          #  type="messages",
          #  show_copy_button=True,
          #  show_copy_all_button=True,
          #  avatar_images=(None, "bot_avatar.png")
        #)
        #chatbot.like(vote, None, None)

        gr.ChatInterface(
          fn=chat,
          examples=["销售二手车的增值税征收办法是什么", "海南离岛免税店应按月进行增值税、消费税纳税申报，在首次进行纳税申报时，应向主管税务机关提供什么资料"],
          concurrency_limit=5,
          title="税务AI小助手"
         # chatbot=chatbot
          )

      demo.queue()
      #demo.launch(share=True)
      demo.launch(server_name="0.0.0.0",server_port=8990)

    elif test_mode == 'batch':
      batch_test()


