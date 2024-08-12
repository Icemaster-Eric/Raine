from typing import Literal
import pickle
import lzma
from tqdm import tqdm
from sentence_transformers.util import semantic_search
from waifumem.models import llm_model, embedding_model, reranking_model
from waifumem.conversation import Conversation


def get_summary(text: str) -> str:
    """Summarizes a conversation

    Args:
        text (str): Conversation as a string

    Returns:
        str: Summary
    """
    return llm_model.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that summarizes conversations. Summarize the following conversation as briefly as possible, without mentioning anything unnecessary:\n{text}"}
    ], temperature=0.3, stop="\n")["choices"][0]["message"]["content"].strip()


def get_topics(text: str) -> str:
    """Returns the topics of a conversation in natural language. May switch to using lmformatenforcer in the future.

    Args:
        text (str): Conversation as a string

    Returns:
        str: Topics
    """
    return llm_model.create_chat_completion([
        {"role": "user", "content": f"You are a smart AI that finds the relevant topics of conversations. Return the topics of the conversation as briefly as possible, without mentioning anything unnecessary besides the topics:\n{text}"}
    ], temperature=0.3, stop="\n")["choices"][0]["message"]["content"].strip()


class Knowledge: # stores knowledge of the model? unsure rn
    def __init__(self):
        pass


class WaifuMem:
    def __init__(self, conversations: list[Conversation] = [], **kwargs):
        self.conversations = []
        self.summaries = []
        self.summary_embeddings = []
        self.topics = []
        self.topic_embeddings = []
        self.settings: dict[Literal["top_k_conv", "min_conv_score", "top_k_msg", "min_msg_score"], int | float] = {
            "top_k_conv": 30,
            "min_conv_score": 0.2,
            "top_k_msg": 100,
            "min_msg_score": 0.25,
        }
        for setting, value in kwargs.items():
            if setting in self.settings:
                self.settings[setting] = value

        for conversation in tqdm(conversations, desc="Generating memory"):
            self.remember(conversation)

    def remember(self, conversation: Conversation):
        self.conversations.append(conversation)
        return # no need for conversation searches right now, I can use it in the future for searching conversations from a long time ago

        # summarize conversation
        summary = conversation.summary or get_summary(conversation.get_text())
        self.summaries.append(summary)
        # embed conversation
        self.summary_embeddings.append(embedding_model.encode(summary, convert_to_tensor=True))

        # get topics of conversation
        topics = conversation.topics or get_topics(conversation.get_text())
        self.topics.append(topics)
        # embed conversation
        self.topic_embeddings.append(embedding_model.encode(topics, convert_to_tensor=True))

    def search_conversation(self, message_embedding, conversation_id: str) -> list[tuple[list[dict[Literal["message", "user", "timestamp"], str | float]], float]]:
        """Semantically searches for relevant messages in a Conversation with large context searches

        Args:
        conversation_id (str): `Conversation.id`

        Returns:
            list: messages from `Conversation.messages`
        """
        conversation = next(conv for conv in self.conversations if conv.id == conversation_id)

        results = semantic_search(message_embedding, conversation.message_ctx_embeddings)[0]

        return sorted([
            (
                conversation.messages_ctx[result["corpus_id"]],
                result["score"]
            ) for result in results if result["score"] > self.settings["min_msg_score"]
        ], reverse=True, key=lambda x: x[1])[:self.settings["top_k_msg"]]

    def search_messages(self):
        # search all messages in all conversations
        pass

    def search_knowledge(self):
        # simply use the reranker
        pass

    def search(self, text: str, top_k: int = 30) -> list[tuple[dict[Literal["message", "user", "timestamp"], str | float], float]]:
        if not self.conversations: # change this once I implement knowledge
            return []

        query = embedding_model.encode(text, prompt_name="query")

        results: list[tuple[list[dict[Literal["message", "user", "timestamp"], str | float]], float]] = []

        for conversation in self.conversations:
            results.extend(self.search_conversation(query, conversation.id))

        results.sort(key=lambda x: x[1], reverse=True)

        scores = reranking_model.predict([
            [text, "\n".join(f"{m['user']}: {m['message']}" for m in result[0])] for result in results
        ])

        # no need to do conversation searches, just do a search on all message triplets
        return [
            (result[0], score) for result, score in sorted(
                zip(results, scores), reverse=True, key=lambda x: x[1]
            )
        ][:top_k]

        if self.summary_embeddings:
            # find relevant conversations by summary (adjust to similarity search based on current conversation's summary?)
            summary_results = semantic_search(query, self.summary_embeddings)[0]

        if self.topic_embeddings:
            # find relevant conversations by topics (adjust to similarity search based on current conversation's topics?)
            topic_results = semantic_search(query, self.topic_embeddings)[0]

        combined_results = [
            (self.conversations[result["corpus_id"]].id, result["score"]) for result in summary_results
        ] + [
            (self.conversations[result["corpus_id"]].id, result["score"]) for result in topic_results
        ]

        conversation_ids = set()
        conversations = []
        for conv_id, score in combined_results:
            if conv_id not in conversation_ids:
                conversations.append((conv_id, score))
                conversation_ids.add(conv_id)

        # filter and sort
        conversations = sorted([
            conv for conv in conversations if conv[1] > self.settings["min_conv_score"]
        ], reverse=True, key=lambda x: x[1])[:self.settings["top_k_conv"]]

        if not conversations:
            return []

        results = []

        for conv_id, score in conversations:
            conversation = next(conv for conv in self.conversations if conv.id == conv_id)

            results.extend(self.search_conversation(query, conversation.id))

        # re-rank results
        scores = reranking_model.predict([
            [text, f"{message[0]['user']}: {message[0]['message']}"] for message in results
        ], convert_to_numpy=False)

        return [result[0] for result in sorted(zip(results, scores), reverse=True, key=lambda x: x[1])][:top_k]

    def save(self, path: str):
        with lzma.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "WaifuMem":
        with lzma.open(path, "rb") as f:
            waifumem = pickle.load(f)

        return waifumem
