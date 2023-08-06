# Content: Embedding engine to create doc embeddings
# Author: Shuai Guo
# Email: shuaiguo0916@hotmail.com
# Date: August, 2023


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import ArxivAPIWrapper
import os


class Embedder:
    """Embedding engine to create doc embeddings."""

    def __init__(self, engine='OpenAI'):
        """Specify embedding model.

        Args:
        --------------
        engine: the embedding model. 
                For a complete list of supported embedding models in LangChain, 
                see https://python.langchain.com/docs/integrations/text_embedding/
        """
        if engine == 'OpenAI':
            self.embeddings = OpenAIEmbeddings()
        
        else:
            raise KeyError("Currently unsupported chat model type!")
        


    def load_n_process_document(self, path):
        """Load and process PDF document.

        Args:
        --------------
        path: path of the paper.
        """

        # Load PDF
        loader = PyMuPDFLoader(path)
        documents = loader.load()

        # Process PDF
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.documents = text_splitter.split_documents(documents)



    def create_vectorstore(self, store_path):
        """Create vector store for doc Q&A.
           For a complete list of vector stores supported by LangChain,
           see: https://python.langchain.com/docs/integrations/vectorstores/

        Args:
        --------------
        store_path: path of the vector store.

        Outputs:
        --------------
        vectorstore: the created vector store for holding embeddings
        """
        if not os.path.exists(store_path):
            print("Embeddings not found! Creating new ones")
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.vectorstore.save_local(store_path)

        else:
            print("Embeddings found! Loaded the computed ones")
            self.vectorstore = FAISS.load_local(store_path, self.embeddings)

        return self.vectorstore
    


    def create_summary(self, llm_engine=None, arxiv_id=None):
        """Create paper summary. If it is an arXiv paper, the 
        summary can be directly fetched from the arXiv. Otherwise, 
        the summary will be created by using LangChain's summarize_chain.

        Args:
        --------------
        llm_engine: backbone large language model.
        arxiv_id: id of the arxiv paper.

        Outputs:
        --------------
        summary: the summary of the paper
        """

        if arxiv_id is None:

            if llm_engine is None:
                raise KeyError("Please provide the arXiv id of the paper! \
                               If this is not an arXiv paper, please specify \
                               a LLM engine to perform summarization.")
            
            elif llm_engine == 'OpenAI':
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.8
                )

            else:
                raise KeyError("Currently unsupported chat model type!")
            
            chain = load_summarize_chain(llm, chain_type="stuff")
            summary = chain.run(self.documents[:2])

        else:
            
            # Retrieve paper metadata
            arxiv = ArxivAPIWrapper()
            summary = arxiv.run(arxiv_id)

            # String manipulation
            summary = summary.replace('{', '(').replace('}', ')')

        return summary