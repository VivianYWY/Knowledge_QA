import os
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
import base64
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config.params as conf
from parser.PDFParse import unstructured_pdf,PyMuPDF_pdf

class RetrieverUse:
    def __init__(self, llm_use, ebd_function, dataset_path):
        self.llm_use = llm_use
        self.ebd_function = ebd_function
        self.dataset_path = dataset_path
        self.dataset_dir = dataset_path + '/files/*'
        self.vector_store_dir = dataset_path + '/database/vector_store/db'
        self.doc_store_dir = dataset_path + '/database/doc_store'
        self.images_output_dir = dataset_path + '/files/imgs'

        self.retriever = None

    def extract_elements_from_dataset(self, dataset_dir):
        output_texts = {}
        output_tables = {}

        # Traverse all files in dataset_dir
        for file in glob.glob(dataset_dir):
            file_format = file.split('.')[-1]
            file_name = file.split('/')[-1].split('.')[0]
            texts = []
            tables = []

            if file_format == 'pdf':  # Process PDF file
                # Get text, tables from pdf
                if "unstructured_pdf" in conf.file_parser:
                    texts, tables = unstructured_pdf(file, self.images_output_dir)
                elif "PyMuPDF_pdf" in conf.file_parser:
                    texts, tables = PyMuPDF_pdf(file, self.images_output_dir)
            else:  # Do not support other format
                raise Exception("目前暂不支持" + file_format)

            if texts:
                # Chunk text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=conf.chunk_size, chunk_overlap=conf.chunk_overlap,
                    separators=["\n\n", r"[\d一二三四五六七八九十百千]+、", "\n"]
                )
                joined_texts = " ".join(texts)
                texts = text_splitter.split_text(joined_texts)

            if tables:
                output_tables[file_name] = tables
            if texts:
                output_texts[file_name] = texts

        return output_texts, output_tables

    # Generate summaries of text elements
    def generate_text_summaries(self, texts, tables, summarize_texts):
        """
        Summarize text elements
        texts: List of str
        tables: List of str
        summarize_texts: Bool to summarize texts
        """
        # Initialize empty summaries
        text_summaries = {}
        table_summaries = {}

        # Apply to text if texts are provided and summarization is requested
        if texts:
            if summarize_texts:
                for key in texts.keys():
                    summaries = []
                    for text in texts[key]:
                        prompt = f'你是负责对文本进行摘要总结的助手，请对下述文本进行摘要总结，目的是通过向量化后的摘要能够准确快速地检索到原始文本。文本: \n{text}\n'
                        query = self.tokenizer.from_list_format([
                            {'text': prompt}
                        ])
                        response, history = self.model.chat(self.tokenizer, query=query, history=None)
                        summaries.append(response)
                    text_summaries[key] = summaries
            else:
                text_summaries = texts

        # Apply to tables if tables are provided
        if tables:
            for key in tables.keys():
                summaries = []
                for table in tables[key]:
                    prompt = f'你是负责对表格进行摘要总结的助手，请对下述表格进行摘要总结，目的是通过向量化后的摘要能够准确快速地检索到原始表格。表格: \n{table}\n'
                    query = self.tokenizer.from_list_format([
                        {'text': prompt}
                    ])
                    response, history = self.model.chat(self.tokenizer, query=query, history=None)
                    summaries.append(response)
                table_summaries[key] = summaries

        return text_summaries, table_summaries

    def encode_image(self, image_path):
        """Getting the base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_summarize(self, img_base64, prompt):
        """Make image summary"""
        query = self.tokenizer.from_list_format([
            {'image': img_base64},
            {'text': prompt}
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def generate_img_summaries(self):
        """
        Generate summaries and base64 encoded strings for images
        """
        # Store base64 encoded images
        img_base64_list = {}

        # Store image summaries
        image_summaries = {}

        # Prompt
        prompt = '你是负责对图片进行摘要总结的助手，请对该图片进行摘要总结，目的是通过向量化后的摘要能够准确快速地检索到原始图片。'

        # Apply to images
        for subdir_name in sorted(os.listdir(self.images_output_dir)):
            tmp_img_base64 = []
            tmp_img_summaries = []

            for img_file in sorted(os.listdir(self.images_output_dir+'/'+subdir_name)):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    img_path = os.path.join(self.images_output_dir+'/'+subdir_name, img_file)
                    base64_image = self.encode_image(img_path)
                    tmp_img_base64.append(base64_image)
                    tmp_img_summaries.append(self.image_summarize(base64_image, prompt))

            img_base64_list[subdir_name] = tmp_img_base64
            image_summaries[subdir_name] = tmp_img_summaries

        return img_base64_list, image_summaries

    def create_retriever(self):

        id_key = "doc_id"

        # Load or create database for vector_store and doc_store
        if os.path.exists(self.vector_store_dir):
            print("------ Load database for vector_store and doc_store...")
            # Load existed database
            ## load vector_store
            vector_store = Chroma(persist_directory=self.vector_store_dir, embedding_function=self.ebd_function)
            ## load doc_store
            doc_store = LocalFileStore(self.doc_store_dir)
        else:
            print("------ Create database for vector_store and doc_store...")
            # Create database from dataset
            ## initialize vector_store
            vector_store = Chroma(persist_directory=self.vector_store_dir, embedding_function=self.ebd_function)
            ## initialize doc_store
            doc_store = LocalFileStore(self.doc_store_dir)

            # Extract elements from dataset (images are stored in local path)
            print("------ Extract elements from files...")
            texts_chunked, tables = self.extract_elements_from_dataset(self.dataset_dir)

            # Get text and table summaries
            print("------ Get text and table summaries...")
            text_summaries, table_summaries = self.generate_text_summaries(
                texts_chunked, tables, conf.is_summarize_texts
            )

            # Get image summaries
            print("------ Get image summaries...")
            img_base64_list, image_summaries = self.generate_img_summaries()

            # Add summaries and raw info to the vector_store and doc_store
            def add_documents(doc_summaries, doc_contents, data_type):
                raw_docs = []
                summary_docs = []
                doc_ids  = []
                for key in doc_summaries.keys():
                    for index, summary in enumerate(doc_summaries[key]):
                        doc_id = str(uuid.uuid4())
                        doc_ids.append(doc_id)
                        summary_docs.append(Document(page_content=summary, metadata={id_key: doc_id, 'type':data_type, 'source':key}))
                        raw_docs.append(doc_contents[key][index])

                # summaries
                vector_store.add_documents(summary_docs)
                # raw info
                doc_store.mset(list(zip(doc_ids, raw_docs)))

            # Add texts, tables, and images
            if text_summaries:
                print("------ Add texts summaries and raw info to the vector_store and doc_store...")
                add_documents(text_summaries, texts_chunked,"text")
            if table_summaries:
                print("------ Add tables summaries and raw info to the vector_store and doc_store...")
                add_documents(table_summaries, tables,"table")
            if image_summaries:
                print("------ Add images summaries and raw info to the vector_store and doc_store...")
                add_documents(image_summaries, img_base64_list,"image")

        # Create the multi-vector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            id_key=id_key,
        )

        # save to disk
        vector_store.persist()

    def use_retriever(self, query):

        raw_texts = []
        raw_tables = []
        raw_imgs = []
        source_files = []
        retrieve_ans = ''

        similar_docs = self.retriever.vector_store.similarity_search_with_relevance_scores(query, k=conf.retrieve_topK)
        related_docs = list(filter(lambda x: x[1] > conf.retrieve_simi, similar_docs))
        print('------ len of related_docs over simi threshold:')
        print(len(related_docs))

        if len(related_docs) != 0:
            for doc, score in related_docs:
                doc_id = doc.metadata.get("doc_id")
                if doc.metadata.get("type") == 'text':
                    raw_texts.append(self.retriever.doc_store.mget([doc_id]))
                elif doc.metadata.get("type") == 'table':
                    raw_tables.append(self.retriever.doc_store.mget([doc_id]))
                elif doc.metadata.get("type") == 'image':
                    raw_imgs.append(self.retriever.doc_store.mget([doc_id]))
                if doc.metadata.get("source") not in source_files:
                    source_files.append(doc.metadata.get("source"))

                tmp = str(score) + '&' + '《' + doc.metadata.get("source") + '》' + '&' + doc.page_content + '&' + self.retriever.doc_store.mget([doc_id])
                if retrieve_ans:
                    retrieve_ans = retrieve_ans + '\n' + tmp
                else:
                    retrieve_ans = tmp

        return raw_texts, raw_tables, raw_imgs, source_files, retrieve_ans
