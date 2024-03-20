import streamlit as st
import prompt_lib as glib
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, Language
import code_snippets
import tiktoken
from langchain.callbacks import StreamlitCallbackHandler

model_options_dict = {
    "anthropic.claude-v2:1": "Claude",
}
st.set_page_config(page_title="Prompt Engineering", layout="wide")

st.title('REXCalibur :sunglasses:  :blue[w4s]  :sunglasses: Hackathon')

model_options = list(model_options_dict)

def get_model_label(model_id):
    return model_options_dict[model_id]
    
if 'memory' not in st.session_state: #see if the memory hasn't been created yet
    st.session_state.memory = glib.get_memory(model_id="anthropic.claude-v2:1") #initialize the memory


if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history




tab1, tab2 = st.tabs(["AI", "LangCHain"])
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Context")
        
        context_list = glib.get_context_list()
        
        selected_context = st.radio(
            "Lab context:",
            context_list,
            #label_visibility="collapsed"
        )
        
        with st.expander("See context"):
            context_for_lab = glib.get_context(selected_context)
            context_text = st.text_area("Context text:", value=context_for_lab, height=350)


    with col2:
        
        
        st.subheader("Prompt & model")
        
        prompt_text = st.text_area("Prompt template text:", height=350)
        
        selected_model = st.radio("Model:", 
            model_options,
            format_func=get_model_label,
            horizontal=True
            #label_visibility="collapsed"
        )
        
        #selected_temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        process_button = st.button("Run", type="primary")


    with col3:
        with st.container(height=1000):

            st.subheader("Result")
            
            #Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
            for message in st.session_state.chat_history: #loop through the chat history
                with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
                    st.markdown(message["text"]) #display the chat content
            if process_button:
                with st.chat_message("user"): #display a user chat message
                    st.markdown(prompt_text) #renders the user's latest message

                
                with st.chat_message("assistant"): #display a bot chat message
                    st_callback = StreamlitCallbackHandler(st.container())
                    st.session_state.chat_history.append({"role":"user", "text":prompt_text}) #append the user's latest message to the chat history
                    response_content = glib.get_text_response(model_id=selected_model, temperature=0.0, template=prompt_text, context=context_text,memory=st.session_state.memory,streaming_callback=st_callback)
                    st_callback._current_thought._container.update(
                        label="",
                        state="complete",
                        expanded=True,
                    )
                    st.session_state.chat_history.append({"role":"assistant", "text":response_content}) #append the bot's latest message to the chat history

with tab2:
        # Streamlit UI
    st.title("Text Splitter")
    st.info("""Split a text into chunks using a **Text Splitter**. Parameters include:

    - `chunk_size`: Max size of the resulting chunks (in either characters or tokens, as selected)
    - `chunk_overlap`: Overlap between the resulting chunks (in either characters or tokens, as selected)
    - `length_function`: How to measure lengths of chunks, examples are included for either characters or tokens
    - The type of the text splitter, this largely controls the separators used to split on
    """)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        chunk_size = st.number_input(min_value=1, label="Chunk Size", value=1000)

    with col2:
        # Setting the max value of chunk_overlap based on chunk_size
        chunk_overlap = st.number_input(
            min_value=1,
            max_value=chunk_size - 1,
            label="Chunk Overlap",
            value=int(chunk_size * 0.2),
        )

        # Display a warning if chunk_overlap is not less than chunk_size
        if chunk_overlap >= chunk_size:
            st.warning("Chunk Overlap should be less than Chunk Length!")

    with col3:
        length_function = st.selectbox(
            "Length Function", ["Characters", "Tokens"]
        )

    splitter_choices = ["RecursiveCharacter", "Character"] + [str(v) for v in Language]

    with col4:
        splitter_choice = st.selectbox(
            "Select a Text Splitter", splitter_choices
        )

    if length_function == "Characters":
        length_function = len
        length_function_str = code_snippets.CHARACTER_LENGTH
    elif length_function == "Tokens":
        enc = tiktoken.get_encoding("cl100k_base")


        def length_function(text: str) -> int:
            return len(enc.encode(text))


        length_function_str = code_snippets.TOKEN_LENGTH
    else:
        raise ValueError

    if splitter_choice == "Character":
        import_text = code_snippets.CHARACTER.format(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function_str
        )

    elif splitter_choice == "RecursiveCharacter":
        import_text = code_snippets.RECURSIVE_CHARACTER.format(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function_str
        )

    elif "Language." in splitter_choice:
        import_text = code_snippets.LANGUAGE.format(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=splitter_choice,
            length_function=length_function_str
        )
    else:
        raise ValueError

    st.info(import_text)

    # Box for pasting text
    doc = st.text_area("Paste your text here:")

    # Split text button
    if st.button("Split Text"):
        # Choose splitter
        if splitter_choice == "Character":
            splitter = CharacterTextSplitter(separator = "\n\n",
                                            chunk_size=chunk_size, 
                                            chunk_overlap=chunk_overlap,
                                            length_function=length_function)
        elif splitter_choice == "RecursiveCharacter":
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap,
                                            length_function=length_function)
        elif "Language." in splitter_choice:
            language = splitter_choice.split(".")[1].lower()
            splitter = RecursiveCharacterTextSplitter.from_language(language,
                                                                    chunk_size=chunk_size,
                                                                    chunk_overlap=chunk_overlap,
                                            length_function=length_function)
        else:
            raise ValueError
        # Split the text
        splits = splitter.split_text(doc)

        # Display the splits
        for idx, split in enumerate(splits, start=1):
            st.text_area(
                f"Split {idx}", split, height=200
            )
