"""
Quality Checker Agent - Validates response quality
"""
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.state import AgentState


class QualityCheckerAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance expert for chatbot responses.
            
            Evaluate the response on:
            1. Relevance to user query (0-1)
            2. Completeness (0-1)
            3. Clarity (0-1)
            4. Helpfulness (0-1)
            
            Return JSON: {{"score": 0.0-1.0, "needs_refinement": true/false, "issues": ["list", "of", "issues"]}}
            
            Score above 0.7 means acceptable quality."""),
            ("user", """
            User Query: {input}
            Response: {response}
            
            Evaluate quality:
            """)
        ])
    
    def process(self, state: AgentState) -> Dict:
        """Check response quality"""
        chain = self.prompt | self.llm
        result = chain.invoke({
            "input": state["user_input"],
            "response": state.get("response", "")
        })
        
        # Parse the response
        import json
        try:
            parsed = json.loads(result.content)
            return {
                "quality_score": parsed.get("score", 0.5),
                "needs_refinement": parsed.get("needs_refinement", False),
                "iteration_count": state.get("iteration_count", 0) + 1
            }
        except json.JSONDecodeError:
            return {
                "quality_score": 0.5,
                "needs_refinement": False,
                "iteration_count": state.get("iteration_count", 0) + 1
            }