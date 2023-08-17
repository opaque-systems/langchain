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
For example, "Giana is good" is sanitized to "PERSON_998 is good".
You must treat the sanitized data as opaque strings, but you can use them as
meaningful entities in the response.
Different sanitized items could be the same entity based on the semantics.
You must keep the sanitized item as is and cannot change it.
The format of sanitized item is "TYPE_ID".
You must not create new sanitized items following the format. For example, you
cannot create "PERSON_997" or "PERSON_999" if "PERSON_997" or "PERSON_999" is
not in the question.

Conversation History: ```{history}```
Context : ```During our recent meeting on February 23, 2023, at 10:30 AM,
John Doe provided me with his personal details. His email is johndoe@example.com
and his contact number is 650-456-7890. He lives in New York City, USA, and
belongs to the American nationality with Christian beliefs and a leaning towards
the Democratic party. He mentioned that he recently made a transaction using his
credit card 4111 1111 1111 1111 and transferred bitcoins to the wallet address
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa. While discussing his European travels, he
noted down his IBAN as GB29 NWBK 6016 1331 9268 19. Additionally, he provided
his website as https://johndoeportfolio.com. John also discussed
some of his US-specific details. He said his bank account number is
1234567890123456 and his drivers license is Y12345678. His ITIN is 987-65-4321,
and he recently renewed his passport,
the number for which is 123456789. He emphasized not to share his SSN, which is
669-45-6789. Furthermore, he mentioned that he accesses his work files remotely
through the IP 192.168.1.1 and has a medical license number MED-123456. ```
Question: ```{question}```
"""


def test_promptguard_llm_wrapper() -> None:
    chain = LLMChain(
        prompt=PromptTemplate.from_template(prompt_template),
        llm=PromptGuardLLMWrapper(llm=OpenAI()),
        memory=ConversationBufferWindowMemory(k=2),
    )

    output = chain.run(
        {
            "question": "Write a text message to remind John to do password reset \
                for his website through his email to stay secure."
        }
    )
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

    pg_chain.invoke(
        {
            "question": "Write a text message to remind John to do password reset\
                 for his website through his email to stay secure.",
            "history": "",
        }
    )
