
from rapidindex.storage.sqllite import SQLiteStorage
s = SQLiteStorage()
docs = s.list_documents()
print(f"Documents in DB: {len(docs)}")
for d in docs:
    full = s.get_document(d.id)
    print(f"  {d.title}: {len(full.sections)} sections")