from typing import Literal
import time
from uuid import uuid4
from copy import deepcopy
from waifumem.models import embedding_model


class Conversation:
    def __init__(self, messages: list[dict[Literal["message", "user", "timestamp"], str | float]] = [], summary: str | None = None, topics: str | None = None):
        """Creates a conversation object that is mutable

        Args:
            messages (list[dict] | None, optional): List of message objects with message (str), user (str) and timestamp (int, epoch seconds) keys. Defaults to None.
        """
        self.id = uuid4().hex
        self.messages = messages
        self.summary = summary
        self.topics = topics

        # cluster messages
        for i, message in enumerate(self.messages):
            if i == 0:
                continue

            if message["user"] == self.messages[i - 1]["user"]:
                if message["timestamp"] - self.messages[i - 1]["timestamp"] > 120:
                    continue

                self.messages[i - 1]["message"] += "\n" + message["message"]
                self.messages[i - 1]["timestamp"] = message["timestamp"]
                self.messages.remove(message)

        self.message_ctx_embeddings = [] # 3 messages embedded at once

        if self.messages:
            for i, message in enumerate(self.messages):
                # horrifyingly disgusting code, but I think it works (add timestamp later?)
                if i == 0:
                    self.message_ctx_embeddings.append(f"{message['user']}: {message['message']}\n{self.messages[i + 1]['user']}: {self.messages[i + 1]['message']}")
                elif i == len(self.messages) - 1:
                    self.message_ctx_embeddings.append(f"{self.messages[i - 1]['user']}: {self.messages[i - 1]['message']}\n{message['user']}: {message['message']}")
                else:
                    self.message_ctx_embeddings.append(f"{self.messages[i - 1]['user']}: {self.messages[i - 1]['message']}\n{message['user']}: {message['message']}\n{self.messages[i + 1]['user']}: {self.messages[i + 1]['message']}")

            self.message_ctx_embeddings = embedding_model.encode(self.message_ctx_embeddings, convert_to_tensor=True)

    def add_message(self, message: str, user: str, timestamp: float | None = None):
        # cluster message
        if self.messages:
            if user == self.messages[-1]["user"]:
                if timestamp - self.messages[-1]["timestamp"] > 120:
                    self.messages[-1] += "\n" + message
                    self.messages[-1]["timestamp"] = timestamp
                    return

        self.messages.append({
            "message": message,
            "user": user,
            "timestamp": timestamp or time.time()
        })

        match len(self.messages):
            case 1:
                self.message_ctx_embeddings.append(embedding_model.encode(f"{user}: {message}", convert_to_tensor=True))
            case 2:
                embedding = embedding_model.encode(f"{self.messages[-2]['user']}: {self.messages[-2]['message']}\n{user}: {message}", convert_to_tensor=True)
                # give the previous message the full context (in this case, it seems a bit weird because both messages have the same embedding, but whatever)
                self.message_ctx_embeddings[-1] = embedding
                self.message_ctx_embeddings.append(embedding)
            case _:
                # give the previous message the full context
                self.message_ctx_embeddings[-1] = embedding_model.encode(f"{self.messages[-3]['user']}: {self.messages[-3]['message']}\n{self.messages[-2]['user']}: {self.messages[-2]['message']}\n{user}: {message}", convert_to_tensor=True)
                self.message_ctx_embeddings.append(embedding_model.encode(f"{self.messages[-2]['user']}: {self.messages[-2]['message']}\n{user}: {message}", convert_to_tensor=True))

    def cut(self, ratio: float = 0.5) -> "Conversation":
        """Cuts the `.messages` list by the ratio and returns a Conversation object with the former slice of the `.messages` list.

        Args:
            ratio (float, optional): The ratio to cut the `.messages` list. Defaults to 0.5.

        Returns:
            Conversation: A new copy of the object with the former slice of the `.messages` list.
        """
        conversation = deepcopy(self)
        split = int(len(self.messages) * ratio)

        conversation.messages = self.messages[:split]
        conversation.message_ctx_embeddings = self.message_ctx_embeddings[:split]

        self.messages = self.messages[split:]
        self.message_ctx_embeddings = self.message_ctx_embeddings[split:]

        return conversation

    @property
    def messages_ctx(self) -> list[list[dict[Literal["message", "user", "timestamp"], str | float]]]:
        messages = []

        for i, message in enumerate(self.messages):
            # horrifyingly disgusting code, but I think it works (add timestamp later?)
            if i == 0:
                messages.append([message, self.messages[i + 1]])
            elif i == len(self.messages) - 1:
                messages.append([self.messages[i - 1], message])
            else:
                messages.append([self.messages[i - 1], message, self.messages[i + 1]])

        return messages

    def get_text(self):
        return "\n".join(f"{message['user']}: {message['message']}" for message in self.messages)
