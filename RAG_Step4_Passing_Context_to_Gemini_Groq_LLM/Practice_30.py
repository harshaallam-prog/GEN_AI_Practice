from Practice_28 import retrieve_context, build_user_prompt, ask_llm

question = "Why does sleep help athletes recover?"
context = retrieve_context(question)
prompt = build_user_prompt(question, context)
print("Using", len(context), "retrieved chunk(s)")

results = {}
for temp in (0.0, 1.0):
    reply = ask_llm(prompt, temperature=temp)
    results[temp] = reply
    print(f"\n--- temperature {temp} | {len(reply.split())} words ---")
    print(reply)

lengths = {t: len(r.split()) for t, r in results.items()}
print("\nWord counts by temperature:", lengths)
