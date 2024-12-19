
import chainlit as cl

from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain

QA_CHAIN_PROMPT = hub.pull('rlm/rag-prompt-mistral')

def load_llm():
    llm = Ollama(
        model="llama3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    return qa_chain

def qa_bot():
    llm = load_llm()
    DB_PATH="vectorstores/db/"
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

# 새로운 채팅 세션이 시작될 때 호출되는 함수를 정의
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(contents="Firing up the research into bot...")
    await msg.send()
    msg.content = "Hi, welcome to the research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

# 사용자가 보낸 메시지를 수신할 때 호출되는 함수
@cl.on_message
async def main(message):
    chain = cl.cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True

    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")

    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()
