from dataclasses import dataclass

@dataclass
class Params:
    dataset: str =  None
    attack_type: str = None
    target_model_type: str = None
    sensitive_attributes: list = None
    missing_nonsensitive_attributes: list = None
    attack_category: str = None
    extra_field_for_attack_category: str = None
    show_case_reports: bool = False
    show_confusion_matrix_visually: bool = False