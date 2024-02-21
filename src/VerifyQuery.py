import re

def remove_names(string):
  filenames = []
  matches = re.findall(r"=.*?]", string)

  # Extract the filenames from the matches.
  for match in matches:
    filename = match[1:-1]
    filenames.append(filename)

  # Remove everything except for the found filenames from the string.
  new_string = string
  for filename in filenames:
    new_string = new_string.replace(filename, "", 1)

  return new_string
  
def validate_brackets(string):
  """
  Validates if the given string contains valid brackets.

  Args:
    string: The string to validate.

  Returns:
    True if the string contains valid brackets, False otherwise.
  """
  
  # Check if the string contains a correct number of brackets.
  if (string.count("[")+string.count("]")) % 4 != 0:
    return False

  # Check if the string contains any brackets inside brackets.
  if not(re.search(r"\[.*\].*\[.*\]", string)):
    return False

  # Check if the string contains any brackets without anything between them.
  if re.search(r"\[\]", string):
    return False

  string_without_filenames = remove_names(string)
  if " " in string_without_filenames:
      return False
          
  return True
  
print(validate_brackets("[-i=\" filename\"]filter[-o=out]"))