import streamlit as st
import pandas as pd
from datetime import datetime
from bot import ChatbotInterface

class MovieChatbotApp:
    def __init__(self):
        # Initialize chatbot only once
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = ChatbotInterface()
        
        self.init_session_state()
        self.setup_ui()

    def init_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chat_df' not in st.session_state:
            st.session_state.chat_df = pd.DataFrame(columns=['Timestamp', 'User', 'Bot'])

    def setup_ui(self):
        st.title("🎬 Arun's IMDB 250 Bot")
        st.markdown("""
        Welcome! I'm your personal guide to the IMDB Top 250 movies.
        Ask me about any movie, search by rank, or get movie details!
        """)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about movies..."):
            # User message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get bot response
            response = st.session_state.chatbot.get_response(prompt)
            
            # Bot message
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Update chat history
            new_row = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'User': prompt,
                'Bot': response
            }
            st.session_state.chat_df = pd.concat([
                st.session_state.chat_df, 
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            # Save to CSV
            st.session_state.chat_df.to_csv('chat_history.csv', index=False)

        # Chat history button
        if st.sidebar.button("Show Chat History"):
            st.sidebar.dataframe(
                st.session_state.chat_df,
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    app = MovieChatbotApp() 