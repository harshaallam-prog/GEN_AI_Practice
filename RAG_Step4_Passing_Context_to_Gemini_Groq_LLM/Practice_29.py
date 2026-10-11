from Practice_28 import collection, store_document, bot

store_document("recovery_and_sleep.txt", """Recovery and Sleep

Sleep is when the body releases growth hormone and repairs tissue. Most athletes perform better when they sleep
seven to nine hours each night.

Gentle activities such as walking, stretching, or swimming speed up recovery by increasing blood flow without
adding much stress to tired muscles.
""")
print("Total chunks now:", collection.count())

test_set = [
    ("How much sleep do athletes need?", "should be answered"),
    ("How does gentle activity help recovery?", "should be answered"),
    ("How do I reset my email password?", "should say I don't know"),
]
for question, expectation in test_set:
    reply = bot.answer(question, show_sources=False)
    refused = "don't know" in reply.lower()
    print(f"\n[{expectation}] {question}\n{reply}\nRefused: {refused}")
