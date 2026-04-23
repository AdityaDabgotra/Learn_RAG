from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self,name,age,grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def get_details(self):
        return self.name
    
    def isPassing(self):
        return self.grade >= 6.0
#Example usage
student1 = Student('Aditya',22,8.8)
print(student1.get_details)

if student1.isPassing():
    print("Student will pass)
else print("Student will fail)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(chunks)
print(len(chunks))
print(chunks[0])