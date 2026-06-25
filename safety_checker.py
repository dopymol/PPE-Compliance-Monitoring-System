def check_ppe_compliance(detected_objects):
    violations = []

    if 'no_helmet' in detected_objects:
        violations.append('Worker without helmet deteted')
    if 'no_gloves' in detected_objects:
        violations.append('Worker without gloves deteted')
    if 'no_goggles' in detected_objects:
        violations.append('Worker without googles deteted')
    if 'no_shoes' in detected_objects:
        violations.append('Worker without safety shoes deteted')

    if violations:
        status = 'Violation'
    else:
        status = "Compliant / No PPE violation detected"

    return status, violations