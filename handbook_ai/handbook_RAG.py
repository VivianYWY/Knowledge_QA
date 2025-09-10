from transformers import TextStreamer, TextIteratorStreamer
import gradio as gr
import time
import config.params as conf
from models.LLM import LLMUse as LLMUse
from models.embedding import EmbeddingUse as EmbeddingUse
from retriever.retriever import RetrieverUse as RetrieverUse
from hit.hit import HitUse
import os



def add_msg(history, message):
    for x in message["files"]:
        print(x)
        if x.split('.')[-1] != 'pdf':
            raise gr.Error("不支持上传非PDF格式文件！")

        msg = gr.ChatMessage(
            role='user',
            content={"path": x}
        )
        history.append(msg)

    if message["text"] is not None:
        msg = gr.ChatMessage(
            role='user',
            content=message["text"]
        )
        history.append(msg)

    return history, gr.MultimodalTextbox(value=None, interactive=False)



def bot(history, is_use):

    query = history[-1]["content"]
    print(query)

    is_hit = False
    theme_check_ans = ''
    retrieve_ans = ''
    generate_ans = ''

    # Check if query is invalid
    if query.replace(" ","") == "":
        history.append(gr.ChatMessage(
            role='assistant',
            content="注意，您的提问为空，请重新提问。"
        ))
        yield history
    else:
        # Check if to use the uploaded file
        if is_use:
            uploaded_filename = ""
            for msg in reversed(history): # traverse chat history to get file info
                if msg["role"] == "user" and isinstance(msg["content"], dict):
                    uploaded_filename = msg["content"]["path"].split(".")[0].split("/")[-1] # Find the latest file uploaded by user
                    break
            # Check if user has uploaded file
            if not uploaded_filename:
                raise gr.Error("还未上传文件！")
            else:
                # Create tmp retriever based on latest file uploaded by user (use filename as pathname)
                retriever_use_tmp = RetrieverUse(model, tokenizer, ebd_function, conf.dataset_path + '/' + uploaded_filename)
                retriever_use_tmp.create_retriever()
                # Retrieve based on query
                raw_texts, raw_tables, raw_imgs, source_files, retrieve_ans = retriever_use_tmp.use_retriever(query)
                # Get answer
                if source_files:
                    # ask LLM based on only raw_texts
                    if raw_texts:
                        prompt = f'你是专业的办公助手，请你结合以下材料回答问题，答案里不允许添加材料范围外的内容，答案里要包含材料中提及的所有时间实体和数字实体，材料：\n{raw_texts}\n问题: \n{query}\n'
                        generate_ans = llm_use.ask('', prompt)

                    source = "来源：" + "，".join(["《" + filename + "》" for filename in source_files])
                    if generate_ans:
                        history.append(gr.ChatMessage(
                            role='assistant',
                            content=generate_ans
                        ))
                        yield history

                    if raw_tables:
                        for tabel in raw_tables:
                            history.append(gr.ChatMessage(
                                role='assistant',
                                content=tabel
                            ))
                            yield history

                    if raw_imgs:
                        for image in raw_imgs:
                            history.append(gr.ChatMessage(
                                role='assistant',
                                content=image
                            ))
                            yield history

                    history.append(gr.ChatMessage(
                        role='assistant',
                        content=source
                    ))
                    yield history

                else:
                    prompt = f'你是专业的办公助手，回答以下问题: \n{query}\n'
                    generate_ans = llm_use.ask('', prompt)

                    source = "来源：本次回答仅依据大模型常识，未依据知识库或用户上传的文件。"
                    history.append(gr.ChatMessage(
                        role='assistant',
                        content=generate_ans
                    ))
                    yield history

                    history.append(gr.ChatMessage(
                        role='assistant',
                        content=source
                    ))
                    yield history

        else:
            # Check if query hit question list
            is_hit, preset_answer = hit_use.question_list_check(query)
            if is_hit:
                # Yield answer with preset_answer
                history.append(gr.ChatMessage(
                    role='assistant',
                    content=preset_answer
                ))
                yield history

                source = "来源：本次回答依据高频问答题库"
                history.append(gr.ChatMessage(
                    role='assistant',
                    content=source
                ))
                yield history

            else:
                # Multi-theme recognize
                themes = "，".join(list(conf.theme_mapping.keys())) + "，其他"
                prompt = f'你是一个分类助手，根据用户的问题，判断用户的提问类型。提问类型分为{themes}。回答时，直接返回提问类型，不要返回其他内容。问题: \n{query}\n'
                theme_check_ans = llm_use.ask('', prompt)

                print("ans of theme check stage:")
                print(theme_check_ans)

                if theme_check_ans not in themes.split("，") or theme_check_ans == "其他":
                    prompt = f'你是专业的办公助手，回答以下问题: \n{query}\n'
                    generate_ans = llm_use.ask('', prompt)
                    
                    source = "来源：本次回答仅依据大模型常识，未依据知识库或用户上传的文件。"
                    history.append(gr.ChatMessage(
                        role='assistant',
                        content=generate_ans
                    ))
                    yield history

                    history.append(gr.ChatMessage(
                        role='assistant',
                        content=source
                    ))
                    yield history

                else:
                    retriever_use = retriever_dict[theme_check_ans]
                    retriever_use.create_retriever()
                    # Retrieve based on query
                    raw_texts, raw_tables, raw_imgs, source_files, retrieve_ans = retriever_use.use_retriever(query)
                    # Get answer
                    if source_files:
                        # ask LLM based on only raw_texts
                        if raw_texts:
                            prompt = f'你是专业的办公助手，请你结合以下材料回答问题，答案里不允许添加材料范围外的内容，答案里要包含材料中提及的所有时间实体和数字实体，材料：\n{raw_texts}\n问题: \n{query}\n'
                            generate_ans = llm_use.ask('', prompt)

                        source = "来源：" + "，".join(["《" + filename + "》" for filename in source_files])
                        if generate_ans:
                            history.append(gr.ChatMessage(
                                role='assistant',
                                content=generate_ans
                            ))
                            yield history

                        if raw_tables:
                            for tabel in raw_tables:
                                history.append(gr.ChatMessage(
                                    role='assistant',
                                    content=tabel
                                ))
                                yield history

                        if raw_imgs:
                            for image in raw_imgs:
                                history.append(gr.ChatMessage(
                                    role='assistant',
                                    content=image
                                ))
                                yield history

                        history.append(gr.ChatMessage(
                            role='assistant',
                            content=source
                        ))
                        yield history

                    else:
                        prompt = f'你是专业的办公助手，回答以下问题: \n{query}\n'
                        generate_ans = llm_use.ask('', prompt)

                        source = "来源：本次回答仅依据大模型常识，未依据知识库或用户上传的文件。"
                        history.append(gr.ChatMessage(
                            role='assistant',
                            content=generate_ans
                        ))
                        yield history

                        history.append(gr.ChatMessage(
                            role='assistant',
                            content=source
                        ))
                        yield history


    with open('log/call_log.txt', 'a+') as f:
        f.write('INVOKE ' + time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime()) + ' ' + query + '&' + str(is_use) + '&' + str(is_hit) + '&' + theme_check_ans + '\n' + retrieve_ans + '\n' + generate_ans)


if __name__ == '__main__':

    print("Start...")

    # Load preset question list
    hit_use = HitUse()
    hit_use.load_question_list(conf.question_list_path)
    print("Finish loading preset question list.")

    # Get embedding function
    embedding_use = EmbeddingUse()
    ebd_function = embedding_use.create_embeddings(conf.Embedding_Model, conf.CUDA_Device)
    print("Finish get embedding function.")

    # Load multi-modal LLM
    llm_use = LLMUse()
    llm_use.load_llm(conf.LLM_Model)
    print("Finish loading multi-modal LLM.")

    # Create retriever for themes
    retriever_dict = {}
    for theme in conf.theme_mapping.keys():
        retriever_use = RetrieverUse(llm_use, ebd_function, conf.dataset_path + '/' + theme)
        retriever_use.create_retriever()
        retriever_dict[theme] = retriever_use
    print("Finish creating retriever for themes.")

    formatted_examples = [{"text": example} for example in conf.examples_dict[list(conf.theme_mapping.keys())[0]]]

    # Create log file
    if not os.path.exists(conf.log_path):
        os.system(r"touch {}".format(conf.log_path))

    # Create gradio demo
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes="title"):
            gr.Markdown(" # <center>员工AI小助手</center>")
            gr.Markdown(" ###### <center>可上传PDF文件问答</center>")

        with gr.Column():
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                bubble_full_width=False,
                type="messages",
                show_copy_button=True,
                show_copy_all_button=True,
                avatar_images=(None, "bot_avatar.png"),
                examples=formatted_examples
            )

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="输入提问，或上传文件...",
                show_label=False
            )
            checkbox = gr.Checkbox(label="使用", info="是否使用上传的文件进行回答")

            clear = gr.Button("清除")
            chat_msg = chat_input.submit(
                add_msg, [chatbot, chat_input], [chatbot, chat_input]
            )
            bot_msg = chat_msg.then(bot, [chatbot, checkbox], chatbot, api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            def add_example(evt: gr.SelectData):
                return evt.value

            def vote(data: gr.LikeData):
                if data.liked:
                    print("You upvoted this response: ")
                    print(data.value)
                    with open('log/call_log.txt', 'a+') as f:
                        f.write("Client upvoted this response." + "\n")
                else:
                    print("You downvoted this response: ")
                    print(data.value)
                    with open('log/call_log.txt', 'a+') as f:
                        f.write("Client downvoted this response." + "\n")

            chatbot.example_select(add_example, None, chat_input)
            chatbot.like(vote, None, None)
            clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(default_concurrency_limit=5)
    demo.launch(share=True, debug=True)
    #demo.launch(server_name="0.0.0.0",server_port=8991)
