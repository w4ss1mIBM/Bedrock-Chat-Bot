import os
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains import ConversationChain

def get_inference_parameters(model_id, temperature): #return a default set of parameters based on the model's provider
    bedrock_model_provider = model_id.split('.')[0] #grab the model provider from the first part of the model id
    
    if (bedrock_model_provider == 'anthropic'): #Anthropic model
        return { #anthropic
            "max_tokens_to_sample": 4000,
            "temperature": temperature, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"] 
           }

    
    else: #Amazon
        #For the LangChain Bedrock implementation, these parameters will be added to the 
        #textGenerationConfig item that LangChain creates for us
        return { 
            "maxTokenCount": 4000, 
            "stopSequences": [], 
            "temperature": temperature, 
            "topP": 0.9 
        }

def get_llm_Memory(model_id):
    

    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id=model_id
        ) 
    
    return llm
def get_llm(model_id, temperature,streaming_callback):
    
    model_kwargs = get_inference_parameters(model_id, temperature)
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id=model_id, #set the foundation model
        streaming=True,
        callbacks=[streaming_callback],
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm

def get_memory(model_id): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm_Memory(model_id)
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=2024) #Maintains a summary of previous messages
    
    return memory
    
def read_file(file_name):
    with open(file_name, "r") as f:
        text = f.read()
     
    return text


def get_context_list():
    return ["Code Explanation", "Code Generatation"]


def get_context(lab):
    if lab == "Prompt engineering basics":
        return read_file("basics.txt")
    if lab == "Summarization":
        return read_file("summarization_content.txt")
    elif lab == "Code":
        return ""
    elif lab == "Advanced techniques: Claude":
        return read_file("summarization_content.txt")


def get_prompt(template, context=None, user_input=None):
    
    prompt_template = PromptTemplate.from_template(template) #this will automatically identify the input variables for the template
    
    if "{context}" not in template:
        prompt = prompt_template.format()
    else:
        prompt = prompt_template.format(context=context) #, user_input=user_input)
    
    return prompt



def get_text_response(model_id, temperature, template, context=None, user_input=None,memory=None,streaming_callback=None): #text-to-text client function
    llm = get_llm(model_id, temperature,streaming_callback)
    
    prompt = get_prompt(template, context, user_input)

    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=prompt) #pass the user message and summary to the model
    
    return chat_response
