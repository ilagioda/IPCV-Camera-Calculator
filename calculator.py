OPERATORS = ['+', '-', '*', '/']

def compute(symbols):
    """
    Solve an arithmetical expression, given an array containing the digits and the operator
    :param symbols: an array of *string* containing the symbols (digits and operators) for the computation
    :return: tuple containing the status code (SUCCESS, ERROR) and the result or the error description
    """

    value = ""
    expression = []
    result = 0

    i = 0
    while i < len(symbols):
        symbol = symbols[i]

        # If the current symbol is part of a number
        if symbol.isdigit() or symbol == '.' or (len(value) == 0 and (symbol == '+' or symbol == '-')):
            # Append the symbol to a string containing the value that's being parsed
            value += symbol
        
        # If the current symbol is a mathematical operator
        elif symbol in OPERATORS or symbol == '=':
            
            # Finish parsing the previous token by converting it to a number
            if len(value) > 0:
                # Parse numerical value and check if any errors
                try:
                    num = float(value)
                    expression.append(num)
                except:
                    return ('ERROR', 'Invalid value: ' + value)

                # Reset the token variable
                value = ""

            else:
                # Unexpected mathematical operator
                return ('ERROR', 'Invalid expression')
            
            # Then append the operator to the list
            expression.append(symbol)

        else:
            # Unexpected symbol
            return ('ERROR', 'Unknown symbol: ' + symbol)

        i += 1      # Increment index

    # Apply all operations in the correct order
    apply_operator('*', expression)
    apply_operator('/', expression)
    apply_operator('+', expression)
    apply_operator('-', expression)

    # Make sure that the expression has been properly solved
    assert len(expression) == 2

    result = expression[0]
    return ('SUCCESS', result)


def apply_operator(operator, expression):
    """
    Find all occurrences of an operator in the provided expression, and replace
    each one with the result obtained by applying the operation to the surrounding values
    :param operator: a value between +, -, * and / which represents the operator to apply
    :param expression: list of math elements (numbers, operators) obtained after parsing
    :return: nothing, the expression list is modified by reference
    """

    # Check that the provided operator is valid
    if not (operator in OPERATORS):
        raise Exception("apply_operator() should only be called with valid operators!")

    i = 1
    while i < len(expression) - 1:

        if (expression[i] == operator):
            op1 = expression[i - 1]
            op2 = expression[i + 1]

            # Apply the operation between the previous and following values
            if (operator == '+'):
                res = op1 + op2
            elif (operator == '-'):
                res = op1 - op2
            elif (operator == '*'):
                res = op1 * op2
            elif (operator == '/'):
                res = op1 / op2

            # Replace the 3 items (op1, operator, op2) with the operation result
            expression[i-1] = res
            del expression[i+1]
            del expression[i]

        i += 1      # Increment index
