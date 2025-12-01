"""
AI Chat Module for F1 Race Analysis.
Provides AI-powered Q&A functionality for race analysis and predictions.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class F1AIChat:
    """
    AI Chat assistant for F1 race analysis.
    Uses Groq API for fast inference.
    """

    # System prompt for F1 context
    SYSTEM_PROMPT = """You are an expert F1 race analyst AI assistant. You help users understand:
- Race strategies and tire management
- Driver and team performance analysis
- Predictions and race outcome analysis
- F1 technical regulations and concepts
- Historical race data and comparisons

Keep your responses concise (under 100 words) but informative.
Use emojis occasionally to make responses engaging (🏎️🏁🔥).
If you don't know something, admit it rather than making up information.
Always be helpful and enthusiastic about F1."""

    def __init__(self):
        """Initialize the AI Chat assistant."""
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.client = None
        self.is_available = False
        self.conversation_history = []
        self.race_context = {}
        
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Groq client."""
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found. AI chat will be disabled.")
            print("   Set your API key: export GROQ_API_KEY=your_key_here")
            return

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.is_available = True
            print("✅ AI Chat initialized successfully")
        except ImportError:
            print("⚠️ groq package not installed. Run: pip install groq")
        except Exception as e:
            print(f"⚠️ Failed to initialize AI Chat: {e}")

    def set_race_context(self, race_info: dict, drivers: list = None, 
                         current_standings: dict = None):
        """
        Set the current race context for more relevant responses.
        
        Args:
            race_info: Dictionary with year, gp, mode, etc.
            drivers: List of driver codes
            current_standings: Current race standings
        """
        self.race_context = {
            "race_info": race_info,
            "drivers": drivers or [],
            "standings": current_standings or {},
        }

    def ask(self, question: str, include_context: bool = True) -> str:
        """
        Ask the AI assistant a question about F1.
        
        Args:
            question: User's question
            include_context: Whether to include current race context
            
        Returns:
            AI response string
        """
        if not self.is_available:
            return self._get_offline_response(question)

        try:
            # Build context message
            context_msg = ""
            if include_context and self.race_context:
                race_info = self.race_context.get("race_info", {})
                if race_info:
                    year = race_info.get("year", "Unknown")
                    gp = race_info.get("gp", "Unknown")
                    mode = race_info.get("mode", "historical")
                    context_msg = f"\n[Current context: {year} {gp} GP, Mode: {mode}]"
                
                standings = self.race_context.get("standings", {})
                if standings:
                    # Sort by position value to get actual top 3
                    top_3 = sorted(standings.items(), key=lambda x: x[1])[:3]
                    if top_3:
                        standings_str = ", ".join([f"P{v}: {k}" for k, v in top_3])
                        context_msg += f"\n[Current standings: {standings_str}]"

            # Prepare messages
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT + context_msg}
            ]
            
            # Add conversation history (last 4 exchanges)
            for msg in self.conversation_history[-8:]:
                messages.append(msg)
            
            # Add current question
            messages.append({"role": "user", "content": question})

            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )

            answer = response.choices[0].message.content

            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            return f"⚠️ AI Error: {str(e)[:50]}... Please try again."

    def _get_offline_response(self, question: str) -> str:
        """
        Provide basic offline responses when AI is not available.
        
        Args:
            question: User's question
            
        Returns:
            Offline response string
        """
        question_lower = question.lower()
        
        # Basic keyword matching for common questions
        if "tyre" in question_lower or "tire" in question_lower:
            return ("🛞 Tyre Strategy Tips:\n"
                   "- SOFT: Fast but wears quickly (15-22 laps)\n"
                   "- MEDIUM: Balanced (25-35 laps)\n"
                   "- HARD: Slower but durable (35-50 laps)")
        
        elif "drs" in question_lower:
            return ("📡 DRS (Drag Reduction System):\n"
                   "Activates when within 1 second of the car ahead in DRS zones. "
                   "Opens rear wing for ~15 km/h speed boost on straights.")
        
        elif "pit" in question_lower or "stop" in question_lower:
            return ("🔧 Pit Stop Strategy:\n"
                   "Most races: 1-2 pit stops\n"
                   "Pit window varies by track and tyre wear.\n"
                   "Typical stop: 2-3 seconds stationary time.")
        
        elif "point" in question_lower or "score" in question_lower:
            return ("🏆 Points System:\n"
                   "P1: 25, P2: 18, P3: 15, P4: 12, P5: 10\n"
                   "P6: 8, P7: 6, P8: 4, P9: 2, P10: 1\n"
                   "+1 for fastest lap (if in top 10)")
        
        else:
            return ("⚠️ AI Chat is offline (GROQ_API_KEY not set).\n"
                   "Basic Q&A available for: tyres, DRS, pit stops, points.\n"
                   "Set GROQ_API_KEY for full AI chat experience!")

    def get_race_summary(self, frame: dict) -> str:
        """
        Generate a brief race summary for the current state.
        
        Args:
            frame: Current race frame data
            
        Returns:
            Race summary string
        """
        if not frame or 'drivers' not in frame:
            return "No race data available."

        drivers = frame['drivers']
        sorted_drivers = sorted(
            drivers.items(),
            key=lambda x: x[1].get('position', 99)
        )

        if len(sorted_drivers) >= 3:
            leader, second, third = sorted_drivers[:3]
            lap = frame.get('lap', 1)
            
            return (f"Lap {lap}: {leader[0]} leads, "
                   f"{second[0]} P2, {third[0]} P3")
        
        return "Race in progress..."

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_quick_tips(self) -> list:
        """
        Get quick tips for users.
        
        Returns:
            List of tip strings
        """
        return [
            "💡 Ask: 'What's the best tyre strategy?'",
            "💡 Ask: 'Who's likely to win?'",
            "💡 Ask: 'Explain DRS'",
            "💡 Ask: 'What's a good pit window?'",
        ]
