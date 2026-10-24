// Prediction form fields. Names, categories and value encodings match the backend's
// Pydantic schema (backend/app/schemas.py) and the training dataset columns.
// Ranges mirror the backend validation limits; the backend re-validates every request.

export const featureSchema = [
  {
    name: 'gender',
    label: 'Gender',
    type: 'select',
    options: [
      { value: 'Female', label: 'Female' },
      { value: 'Male', label: 'Male' },
      { value: 'Other', label: 'Other' },
    ],
  },
  { name: 'age', label: 'Age', type: 'number', unit: 'years', min: 0, max: 120, step: 'any' },
  {
    name: 'hypertension',
    label: 'Hypertension',
    type: 'binary',
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  {
    name: 'heart_disease',
    label: 'Heart disease',
    type: 'binary',
    options: [
      { value: 0, label: 'No' },
      { value: 1, label: 'Yes' },
    ],
  },
  {
    name: 'smoking_history',
    label: 'Smoking history',
    type: 'select',
    options: [
      { value: 'never', label: 'Never' },
      { value: 'former', label: 'Former' },
      { value: 'current', label: 'Current' },
      { value: 'not current', label: 'Not current' },
      { value: 'ever', label: 'Ever' },
      { value: 'No Info', label: 'No information' },
    ],
  },
  { name: 'bmi', label: 'BMI', type: 'number', unit: 'kg/m²', min: 10, max: 100, step: 'any' },
  { name: 'HbA1c_level', label: 'HbA1c level', type: 'number', unit: '%', min: 3, max: 20, step: 'any' },
  {
    name: 'blood_glucose_level',
    label: 'Blood glucose level',
    type: 'number',
    unit: 'mg/dL',
    min: 40,
    max: 600,
    step: 'any',
  },
];

export function displayValue(field, value) {
  if (value === undefined || value === null || value === '') return '—';
  const option = field.options?.find((o) => o.value === value);
  if (option) return option.label;
  return field.unit ? `${value} ${field.unit}` : String(value);
}

// Returns { fieldName: message } for invalid fields; empty object when the form is valid.
export function validateForm(values) {
  const errors = {};
  for (const field of featureSchema) {
    const raw = values[field.name];
    if (raw === '' || raw === undefined || raw === null) {
      errors[field.name] = 'Required';
      continue;
    }
    if (field.type === 'number') {
      const n = Number(raw);
      if (!Number.isFinite(n)) errors[field.name] = 'Enter a number';
      else if (n < field.min || n > field.max) errors[field.name] = `Must be between ${field.min} and ${field.max}`;
    } else if (!field.options.some((o) => String(o.value) === String(raw))) {
      errors[field.name] = 'Select a valid option';
    }
  }
  return errors;
}

// Converts form state (strings) into the typed payload expected by POST /api/predict.
export function toPayload(values) {
  return Object.fromEntries(
    featureSchema.map((f) => {
      const raw = values[f.name];
      if (f.type === 'number' || f.type === 'binary') return [f.name, Number(raw)];
      return [f.name, raw];
    }),
  );
}
