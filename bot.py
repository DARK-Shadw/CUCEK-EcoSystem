import streamlit as st

class Chatbot:
    def __init__(self):
        """Initialize the chatbot with an empty message history."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def display_messages(self):
        """Render the chat messages in the Streamlit app."""
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    def send_message(self, user_input):
        """Process user input and generate a bot response."""
        # Display user message
        with st.chat_message('user'):
            st.markdown(user_input)
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        # Generate bot response (echoing user input in this example)
        response = f'You said: {user_input}'
        with st.chat_message('bot'):
            st.markdown(response)
        st.session_state.messages.append({'role': 'bot', 'content': response})

def main():
    st.title('Simple Chatbot')
    st.markdown('Hello! How can I assist you today?')

    chatbot = Chatbot()
    chatbot.display_messages()

    user_input = st.chat_input('Enter your message:')
    if user_input:
        chatbot.send_message(user_input)

if __name__ == '__main__':
    main()
