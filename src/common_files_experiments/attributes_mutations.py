from typing import List, Dict, Union
from dataclasses import dataclass

@dataclass
class Mutation:
    field_identified: str
    value_in_field: str
    action: str  # "replace", "remove", or "add"
    replacement_dict: Dict = None  # Used for "replace" or "add"

def mutate_attributes(attributes: List[Dict], mutations: List[Mutation]) -> List[Dict]:
    for mutation in mutations:
        if mutation.action == "replace":
            for idx, attribute in enumerate(attributes):
                if attribute.get(mutation.field_identified) == mutation.value_in_field:
                    attributes[idx] = mutation.replacement_dict
        elif mutation.action == "remove":
            attributes = [
                attribute for attribute in attributes
                if attribute.get(mutation.field_identified) != mutation.value_in_field
            ]
        elif mutation.action == "add":
            if mutation.replacement_dict:
                attributes.append(mutation.replacement_dict)
    return attributes
