from Practice_28 import bot

question = "How do training, nutrition and sleep work together for performance?"
print("Stricter retrieval = fewer, better chunks. Comparing minimum-similarity settings:")
for cutoff in (0.2, 0.5, 0.8):
    bot.min_similarity = cutoff
    chunks = bot.retrieve(question, n_results=6)
    print(f"\nmin_similarity={cutoff} -> {len(chunks)} chunk(s)")
    for c in chunks:
        print(f"   {c['similarity']} {c['source']} #{c['chunk_index']}")
    print("Answer:", bot.answer(question, n_results=6, show_sources=False)[:200].replace("\n", " "))
