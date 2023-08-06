# Content: Class definition of interactive chatbot system 
# Author: Shuai Guo
# Email: shuaiguo0916@hotmail.com
# Date: August, 2023


from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from abc import ABC, abstractmethod


class Chatbot(ABC):
    """Class definition for a single chatbot with memory, created with LangChain."""
    
    
    def __init__(self, engine):
        """Initialize the large language model and its associated memory.
        The memory can be an LangChain emory object, or a list of chat history.

        Args:
        --------------
        engine: the backbone llm-based chat model.
                "OpenAI" stands for OpenAI chat model;
                Other chat models are also possible in LangChain, 
                see https://python.langchain.com/en/latest/modules/models/chat/integrations.html
        """
        
        # Instantiate llm
        if engine == 'OpenAI':
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.8
            )

        else:
            raise KeyError("Currently unsupported chat model type!")


    @abstractmethod
    def instruct(self):
        """Determine the context of chatbot interaction. 
        """
        pass


    @abstractmethod
    def step(self):
        """Action produced by the chatbot. 
        """
        pass
        

    @abstractmethod
    def _specify_system_message(self):
        """Prompt engineering for chatbot.
        """       
        pass
    



class JournalistBot(Chatbot):
    """Class definition for the journalist bot, created with LangChain."""
    
    def __init__(self, engine):
        """Setup journalist bot.
        
        Args:
        --------------
        engine: the backbone llm-based chat model.
                "OpenAI" stands for OpenAI chat model;
                Other chat models are also possible in LangChain, 
                see https://python.langchain.com/en/latest/modules/models/chat/integrations.html
        """
        
        # Instantiate llm
        super().__init__(engine)
        
        # Instantiate memory
        self.memory = ConversationBufferMemory(return_messages=True)


    def instruct(self, topic, abstract):
        """Determine the context of journalist chatbot. 
        
        Args:
        ------
        topic: the topic of the paper
        abstract: the abstract of the paper
        """
        
        self.topic = topic
        self.abstract = abstract
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])
        
        # Create conversation chain
        self.conversation = ConversationChain(memory=self.memory, prompt=prompt, 
                                              llm=self.llm, verbose=False)
        

    def step(self, prompt):
        """Journalist chatbot asks question. 
        
        Args:
        ------
        prompt: Previos answer provided by the author bot.
        """
        response = self.conversation.predict(input=prompt)
        
        return response
        


    def _specify_system_message(self):
        """Specify the behavior of the journalist chatbot.


        Outputs:
        --------
        prompt: instructions for the chatbot.
        """       
        
        # Compile bot instructions 
        prompt = f"""You are a technical journalist interested in {self.topic}, 
        Your task is to distill a recently published scientific paper on this topic through
        an interview with the author, which is played by another chatbot.
        Your objective is to ask comprehensive and technical questions 
        so that anyone who reads the interview can understand the paper's main ideas and contributions, 
        even without reading the paper itself. 
        You're provided with the paper's summary to guide your initial questions.
        You must keep the following guidelines in mind:
        - Focus exclusive on the technical content of the paper.
        - Avoid general questions about {self.topic}, focusing instead on specifics related to the paper.
        - Only ask one question at a time.
        - Feel free to ask about the study's purpose, methods, results, and significance, 
        and clarify any technical terms or complex concepts. 
        - Your goal is to lead the conversation towards a clear and engaging summary.
        - Do not include any prefixed labels like "Interviewer:" or "Question:" in your question.
        
        [Abstract]: {self.abstract}"""
        
        
        return prompt



class AuthorBot(Chatbot):
    """Class definition for the author bot, created with LangChain."""
    
    def __init__(self, engine, vectorstore, debug=False):
        """Select backbone large language model, as well as instantiate 
        the memory for creating language chain in LangChain.
        
        Args:
        --------------
        engine: the backbone llm-based chat model.
        vectorstore: embedding vectors of the paper.
        """
        
        # Instantiate llm
        super().__init__(engine)
        
        # Instantiate memory
        self.chat_history = []
        
        # Instantiate embedding index
        self.vectorstore = vectorstore

        self.debug = debug
        
        
        
    def instruct(self, topic):
        """Determine the context of author chatbot. 
        
        Args:
        -------
        topic: the topic of the paper.
        """

        # Specify topic
        self.topic = topic
        
        # Define prompt template
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self._specify_system_message()),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # Create conversation chain
        self.conversation_qa = ConversationalRetrievalChain.from_llm(llm=self.llm, verbose=self.debug,
                                                                     retriever=self.vectorstore.as_retriever(
                                                                         search_kwargs={"k": 3}),
                                                                    chain_type="stuff", return_source_documents=True,
                                                                    combine_docs_chain_kwargs={'prompt': qa_prompt})

        
        
    def step(self, prompt):
        """Author chatbot answers question. 
        
        Args:
        ------
        prompt: question raised by journalist bot.

        Outputs:
        ------
        answer: the author bot's answer
        source_documents: documents that author bot used to answer questions
        """
        response = self.conversation_qa({"question": prompt, "chat_history": self.chat_history})
        self.chat_history.append((prompt, response["answer"]))
        
        return response["answer"], response["source_documents"]
        
        
        
    def _specify_system_message(self):
        """Specify the behavior of the author chatbot.


        Outputs:
        --------
        prompt: instructions for the chatbot.
        """       
        
        # Compile bot instructions 
        prompt = f"""You are the author of a recently published scientific paper on {self.topic}.
        You are being interviewed by a technical journalist who is played by another chatbot and
        looking to write an article to summarize your paper.
        Your task is to provide comprehensive, clear, and accurate answers to the journalist's questions.
        Please keep the following guidelines in mind:
        - Try to explain complex concepts and technical terms in an understandable way, without sacrificing accuracy.
        - Your responses should primarily come from the relevant content of this paper, 
        which will be provided to you in the following, but you can also use your broad knowledge in {self.topic} to 
        provide context or clarify complex topics. 
        - Remember to differentiate when you are providing information directly from the paper versus 
        when you're giving additional context or interpretation. Use phrases like 'According to the paper...' for direct information, 
        and 'Based on general knowledge in the field...' when you're providing additional context.
        - Only answer one question at a time. Ensure that each answer is complete before moving on to the next question.
        - Do not include any prefixed labels like "Author:", "Interviewee:", Respond:", or "Answer:" in your answer.
        """
        
        prompt += """Given the following context, please answer the question.
        
        {context}"""
        
        return prompt