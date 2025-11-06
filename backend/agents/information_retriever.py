"""
Information Retrieval Agent - Gathers relevant information based on intent
"""
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.state import AgentState


class InformationRetrieverAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an information retrieval specialist.
            Based on the user's query and detected intent, extract and organize key information needed.
            
            Provide:
            1. Key facts or data points
            2. Relevant context
            3. Any clarifications needed
            
            Be concise and structured."""),
            ("user", """
            User Query: {input}
            Intent: {intent}
            Context: {context}
            
            Retrieve relevant information:
            """)
        ])
    
    def process(self, state: AgentState) -> Dict:
        """Retrieve relevant information"""
        chain = self.prompt | self.llm
        result = chain.invoke({
            "input": state["user_input"],
            "intent": state.get("intent", "unknown"),
            "context": state.get("context", {})
        })
        
        return {
            "retrieved_info": result.content
        }