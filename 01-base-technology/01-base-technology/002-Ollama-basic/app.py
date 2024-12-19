
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear() # 객체 초기화
    st.session_state["messages"] = [] # 이전 대화 메세지 초기화
    st.session_state["user_input"] = "" # 사용자 입력 필드 초기화

    for file in st.session_state["file_uploader"]:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False) as tf: # delete=False는 프로세스 종료 후에도 파일이 유지되도록 설정
            tf.write(file.getbuffer()) # 업로드된 파일의 바이너리 데이터를 임시 파일에 씀
            file_path = tf.name # 임시 파일의 경로를 저장

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
      if len(st.session_state) == 0:
          st.session_state["messages"] = []
          st.session_state["assistant"] = ChatPDF()

      st.header("ChatPDF")

      st.subheader("Upload a document")
      st.file_uploader(
          "Upload document",
          type=["pdf"],
          key="file_uploader",
          on_change=read_and_save_file,
          label_visibility="collapsed",
          accept_multiple_files=True,
      )

      st.session_state["ingestion_spinner"] = st.empty()

      display_messages()
      st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
    print("a")
