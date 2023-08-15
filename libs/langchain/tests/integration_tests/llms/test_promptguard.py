import langchain.utilities.promptguard as pgf
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.llms.promptguard import PromptGuardLLMWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap

prompt_template = """
As an AI assistant, you will answer questions according to given context.

Important PII data is sanitized in the question.
For example, "Giana is good" is sanitized to "PERSON_999 is good".
You must treat the sanitized data as opaque strings, but you can use them as
meaningful entities in the response.
Different sanitized items could be the same entity based on the semantics.
You must keep the sanitized item as is and cannot change it.
The format of sanitized item is "TYPE_ID".
You must not create new sanitized items following the format. For example, you
cannot create "PERSON_1000" or "PERSON_998" if "PERSON_1000" or "PERSON_998" is
not in the question.

Conversation History: ```{history}```
Context : ```Mr. Carl Smith is a 31-year-old man who has been experiencing
homelessness on and off for all his adult life. Mr. Smith says he is about
5’5” and weighs approximately 129 lbs. He presents as
very thin, typically wearing a clean white undershirt
and loose-fitting khaki shorts at interviews.
His brown hair is disheveled and dirty looking, and
he constantly fidgets and shakes his hand or
knee during interviews. Despite his best efforts, Carl is a poor historian. ```
Question: ```{question}```
"""


def test_promptguard_llm_wrapper() -> None:
    chain = LLMChain(
        prompt=PromptTemplate.from_template(prompt_template),
        llm=PromptGuardLLMWrapper(llm=OpenAI()),
        memory=ConversationBufferWindowMemory(k=2),
    )

    output = chain.run({"question": """How high is he? """})
    assert isinstance(output, str)


def test_promptguard_functions() -> None:
    prompt = (PromptTemplate.from_template(prompt_template),)
    llm = OpenAI()
    pg_chain = (
        pgf.sanitize
        | RunnableMap(
            {
                "response": (lambda x: x["sanitized_input"])  # type: ignore
                | prompt
                | llm
                | StrOutputParser(),
                "secure_context": lambda x: x["secure_context"],
            }
        )
        | (lambda x: pgf.desanitize(x["response"], x["secure_context"]))
    )

    pg_chain.invoke({"question": "How high is he?", "history": ""})
