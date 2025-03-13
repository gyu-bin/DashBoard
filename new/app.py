import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 페이지 설정
st.set_page_config(
    page_title="PDF 기반 Q&A 시스템",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 기반 Q&A 시스템")
st.markdown("업로드한 PDF 문서를 기반으로 질문에 답변하는 시스템입니다.")

# API 키 입력 (Streamlit에서 getpass 대신 text_input 사용)
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

###########################################
# PDF 업로드 및 처리 함수 (Streamlit 버전)
###########################################
def upload_and_process_pdf():
    """
    사용자가 PDF 파일을 업로드하고 이를 처리하는 함수 (Streamlit 버전)
    """
    uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요...", type=["pdf"])
    if uploaded_file is None:
        st.warning("파일이 업로드되지 않았습니다.")
        return None, None

    filename = uploaded_file.name

    # 업로드된 파일을 임시 파일에 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name
    temp_file.close()

    st.info(f"'{filename}' 파일이 업로드되었습니다. 처리를 시작합니다...")

    # PDF 파일 로드
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    st.info(f"PDF에서 {len(documents)} 페이지를 로드했습니다.")

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    st.info(f"문서를 {len(chunks)} 개의 청크로 분할했습니다.")

    # 임시 파일 삭제
    os.unlink(temp_path)

    return chunks, filename

document_chunks, pdf_filename = upload_and_process_pdf()
if document_chunks is None:
    st.stop()

###########################################
# 벡터 스토어 생성 함수
###########################################
def create_vector_store(chunks):
    if chunks is None or len(chunks) == 0:
        st.error("처리할 문서 청크가 없습니다.")
        return None

    st.info("임베딩 모델을 초기화하고 벡터 스토어를 생성합니다...")
    # 임베딩 모델 선택
    option = st.radio("임베딩 모델 선택", ("OpenAI 임베딩", "HuggingFace 임베딩"))
    if option == "OpenAI 임베딩":
        embeddings = OpenAIEmbeddings()
        st.info("OpenAI 임베딩 모델을 사용합니다.")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        st.info("HuggingFace 임베딩 모델을 사용합니다.")

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success("벡터 스토어 생성이 완료되었습니다!")
        return vector_store
    except Exception as e:
        st.error(f"벡터 스토어 생성 중 오류가 발생했습니다: {e}")
        return None

vector_store = create_vector_store(document_chunks)
if vector_store is None:
    st.stop()

###########################################
# RAG 체인 설정 함수
###########################################
def setup_rag_chain(vector_store):
    if vector_store is None:
        st.error("벡터 스토어가 없어 RAG 체인을 설정할 수 없습니다.")
        return None

    st.info("RAG 체인을 구성합니다...")
    # 모델 선택
    model_choice = st.radio("사용할 모델 선택", ("gpt-3.5-turbo", "gpt-4"))
    if model_choice == "gpt-4":
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        st.info("GPT-4 모델을 사용합니다.")
    else:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        st.info("GPT-3.5-turbo 모델을 사용합니다.")

    # 대화 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 검색기 설정 (상위 3개 청크 검색)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 프롬프트 템플릿 설정
    qa_template = """
    당신은 PDF 문서의 내용에 기반하여 질문에 답변하는 도우미입니다.

    주어진 정보만을 사용하여 질문에 답변하세요. 정보가 충분하지 않다면, "주어진 문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.

    항상 문서의 내용에 충실하게 답변하고, 추측하지 마세요.

    질문: {question}

    관련 문서 내용:
    {context}

    답변:
    """
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    # RAG 체인 구성
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    st.success("RAG 체인 구성이 완료되었습니다!")
    return qa_chain

qa_chain = setup_rag_chain(vector_store)
if qa_chain is None:
    st.stop()

###########################################
# 대화형 인터페이스 (Streamlit 방식)
###########################################
st.header(f"{pdf_filename} 문서와 대화하기")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("질문을 입력하세요:")
if st.button("전송"):
    if question.strip() == "":
        st.warning("질문을 입력해주세요.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("답변을 생성 중입니다..."):
            response = qa_chain({"question": question})
            answer = response.get("answer", "답변 생성 실패")
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.success("답변 생성 완료!")

# 대화 이력 표시
if st.session_state.chat_history:
    st.subheader("대화 이력")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**질문:** {msg['content']}")
        else:
            st.markdown(f"**답변:** {msg['content']}")

###########################################
# 대화 이력 시각화 함수 (옵션)
###########################################
def visualize_conversation_history(qa_chain):
    if qa_chain is None or not hasattr(qa_chain, 'memory') or qa_chain.memory is None:
        st.error("대화 이력을 가져올 수 없습니다.")
        return

    chat_history = qa_chain.memory.chat_memory.messages
    if not chat_history:
        st.info("아직 대화 이력이 없습니다.")
        return

    st.subheader("메모리 기반 대화 이력")
    for i, message in enumerate(chat_history):
        if hasattr(message, 'type') and message.type == 'human':
            st.markdown(f"**질문 {i//2 + 1}:** {message.content}")
        elif hasattr(message, 'type') and message.type == 'ai':
            st.markdown(f"**답변 {i//2 + 1}:** {message.content}")
            st.markdown("---")

# 대화 이력 시각화 버튼
if st.button("메모리 대화 이력 보기"):
    visualize_conversation_history(qa_chain)

###########################################
# 다중 PDF 처리 (옵션)
###########################################
st.header("여러 PDF 파일 업로드 및 처리 (옵션)")
multi_pdf_uploaded = st.file_uploader("여러 PDF 파일을 업로드하세요 (여러 파일 선택 가능)", type=["pdf"], accept_multiple_files=True)
if multi_pdf_uploaded:
    all_chunks = []
    filenames = []
    for file in multi_pdf_uploaded:
        if not file.name.lower().endswith('.pdf'):
            st.warning(f"'{file.name}'은(는) PDF 파일이 아닙니다. 건너뜁니다.")
            continue

        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(file.read())
        temp_path = temp_file.name
        temp_file.close()

        st.info(f"'{file.name}' 파일을 처리합니다...")
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            st.info(f"- {len(documents)} 페이지 로드됨.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_documents(documents)
            st.info(f"- {len(chunks)} 청크로 분할됨.")

            # 각 청크에 파일명 추가
            for chunk in chunks:
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = file.name

            all_chunks.extend(chunks)
            filenames.append(file.name)
        except Exception as e:
            st.error(f"'{file.name}' 처리 중 오류 발생: {e}")
        finally:
            os.unlink(temp_path)

    if all_chunks:
        st.success(f"총 {len(all_chunks)} 청크가 {len(filenames)}개의 PDF에서 추출됨.")
        # 벡터스토어 생성 (옵션)
        try:
            embeddings = OpenAIEmbeddings()
            multi_vector_store = FAISS.from_documents(all_chunks, embeddings)
            st.success("여러 PDF 벡터 스토어 생성 완료!")
            multi_qa_chain = setup_rag_chain(multi_vector_store)
            st.header("여러 PDF 문서와의 대화")
            multi_question = st.text_input("여러 PDF에 대한 질문 입력:", key="multi_q")
            if st.button("전송 (여러 PDF)"):
                if multi_question.strip() == "":
                    st.warning("질문을 입력해주세요.")
                else:
                    with st.spinner("답변 생성 중..."):
                        multi_response = multi_qa_chain({"question": multi_question})
                        multi_answer = multi_response.get("answer", "답변 생성 실패")
                    st.success("답변 생성 완료!")
                    st.markdown(f"**답변:** {multi_answer}")
        except Exception as e:
            st.error(f"다중 PDF 벡터 스토어 생성 중 오류: {e}")
    else:
        st.warning("처리된 PDF 문서가 없습니다.")

st.markdown("---")
st.caption("© 2023 학생 성적 관리 대시보드 | Streamlit으로 제작되었습니다")
