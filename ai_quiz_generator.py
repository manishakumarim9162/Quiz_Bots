"""
🤖 AI QUIZ GENERATOR MODULE
Uses Google Gemini API to generate quizzes from topics/descriptions
"""

import json
import logging
import os
from datetime import datetime
import google.generativeai as genai
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

class AIQuizGenerator:
    """
    🎯 Generate quizzes using AI based on topics
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro') if GOOGLE_API_KEY else None
        
    def is_available(self) -> bool:
        """Check if AI generation is available"""
        return self.model is not None
    
    async def generate_quiz_from_topic(
        self, 
        topic: str, 
        num_questions: int = 5, 
        difficulty: str = "medium"
    ) -> Optional[Dict]:
        """
        Generate quiz questions based on a topic
        
        Args:
            topic: Topic/subject for the quiz
            num_questions: Number of questions to generate (3-10)
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            Dict with quiz data or None if failed
        """
        
        if not self.model:
            logger.error("AI model not initialized")
            return None
        
        # Validate inputs
        num_questions = max(3, min(10, num_questions))  # 3-10 questions
        difficulty = difficulty.lower() if difficulty else "medium"
        
        if difficulty not in ["easy", "medium", "hard"]:
            difficulty = "medium"
        
        prompt = f"""
        Generate a quiz with {num_questions} multiple choice questions about: "{topic}"
        Difficulty level: {difficulty}
        
        Return ONLY valid JSON (no markdown, no code blocks) with this exact structure:
        {{
            "title": "Quiz Title",
            "description": "Brief description",
            "questions": [
                {{
                    "text": "Question text?",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct_answer": "Option 1",
                    "explanation": "Why this is correct...",
                    "pre_message": "Context or hint (optional)"
                }}
            ]
        }}
        
        Requirements:
        - Each question must have 4 options
        - Exactly one correct answer matching an option
        - Clear, concise explanations
        - Appropriate difficulty level
        - No duplicate questions
        - Return only valid JSON
        """
        
        try:
            logger.info(f"🤖 Generating AI quiz: topic='{topic}', questions={num_questions}, difficulty={difficulty}")
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ```
                
            response_text = response_text.strip()
            
            # Parse JSON
            quiz_data = json.loads(response_text)
            
            # Validate structure
            if not self._validate_quiz_data(quiz_data, num_questions):
                logger.error("Generated quiz validation failed")
                return None
            
            logger.info(f"✅ AI quiz generated successfully: {len(quiz_data['questions'])} questions")
            return quiz_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Response was: {response_text[:200]}")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating quiz: {e}")
            return None
    
    def _validate_quiz_data(self, data: Dict, expected_questions: int) -> bool:
        """Validate generated quiz structure"""
        try:
            if not isinstance(data, dict):
                return False
            
            required_keys = ["title", "description", "questions"]
            if not all(key in data for key in required_keys):
                return False
            
            questions = data["questions"]
            if not isinstance(questions, list) or len(questions) != expected_questions:
                return False
            
            for q in questions:
                if not isinstance(q, dict):
                    return False
                
                req_keys = ["text", "options", "correct_answer", "explanation"]
                if not all(key in q for key in req_keys):
                    return False
                
                if not isinstance(q["options"], list) or len(q["options"]) < 3:
                    return False
                
                if q["correct_answer"] not in q["options"]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def generate_from_description(
        self, 
        description: str, 
        num_questions: int = 5
    ) -> Optional[Dict]:
        """
        Generate quiz from detailed description/content
        
        Args:
            description: Detailed content to create quiz from
            num_questions: Number of questions
            
        Returns:
            Quiz data dict or None
        """
        
        if not self.model:
            return None
        
        num_questions = max(3, min(10, num_questions))
        
        prompt = f"""
        Create a {num_questions}-question quiz based on this content:
        
        "{description}"
        
        Return ONLY valid JSON (no markdown):
        {{
            "title": "Quiz title derived from content",
            "description": "Brief description",
            "questions": [
                {{
                    "text": "Question?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "Explanation",
                    "pre_message": ""
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            quiz_data = json.loads(response_text)
            
            if self._validate_quiz_data(quiz_data, num_questions):
                logger.info(f"✅ Quiz generated from description: {len(quiz_data['questions'])} questions")
                return quiz_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating from description: {e}")
            return None


# Singleton instance
ai_generator = AIQuizGenerator()


def get_ai_generator() -> AIQuizGenerator:
    """Get AI Generator instance"""
    return ai_generator
