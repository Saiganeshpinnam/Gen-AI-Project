def is_low_confidence(source_docs):
    return len(source_docs) == 0


def route_query(result):
    if is_low_confidence(result["source_documents"]):
        return "ESCALATE"
    return "ANSWER"
