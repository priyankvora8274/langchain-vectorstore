from langchain_classic.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.schema import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Texas is one of the largest and most popular states in the United States. \
It is located in the southern part of the country and is known for its rich history, diverse culture, and strong economy.",
    metadata = {"State":"Texas"}
)

doc2 = Document(
    page_content="California is a beautiful and economically important state located on the western coast of the United States. \
Its capital is Sacramento, and its largest city is Los Angeles. California is famous for its beaches, mountains, national parks, and pleasant weather in many areas",
    metadata = {"State":"California"}
)

doc3 = Document(
    page_content="New York is one of the most famous states in the United States, located in the northeastern part of the country. \
Its capital is Albany, while New York City is its largest and most well-known city.",
    metadata = {"State":"New York"}
)

chromadb = Chroma(
    embedding_function= OpenAIEmbeddings(),
    persist_directory= 'chroma_db',
    collection_name='usa_state_info'
)

chromadb.add_documents([doc1,doc2,doc3])

db_data = chromadb.get(include=['embeddings','metadatas','documents'])

result = chromadb.similarity_search(query="which one is located in south?", k=1) 
# print(result)

result1 = chromadb.similarity_search_with_score(query="which one is located in south?", k=1 ) 
print(result1)

chromadb.update_document(document_id="")
