"""Module for parsing and evaluating arithmetic expressions"""

OPERATORS = ['+', '-', '*', '/']

def compute(symbols):
    """
    Solve an arithmetical expression, given an array containing the digits and the operator
    :param symbols: an array of *strings* with the symbols (digits and operators)
    :return: tuple containing the outcome (SUCCESS/ERROR) and the result or the error description
    """
    if not symbols or len(symbols) == 0:
        return ('ERROR', 'Empty expression')

    # The last symbol of an expression must be the '=' sign
    assert symbols[-1] == '=', "The expression must be terminated by '='"

    value = ""
    expression = []
    result = 0

    i = 0
    while i < len(symbols):
        symbol = symbols[i]

        # If the current symbol is part of a number
        if symbol.isdigit() or symbol == '.' or (len(value) == 0 and symbol in ('+', '-')):
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
                except ValueError:
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
    apply_operators(['*', '/'], expression)
    apply_operators(['+', '-'], expression)

    # Make sure that the expression has been properly solved
    assert len(expression) == 2, "Unable to solve the expression"

    result = expression[0]
    return ('SUCCESS', result)


def apply_operators(operators, expression):
    """
    Find all occurrences of the provided operators in the provided expression, and replace
    each one with the result obtained by applying the operation to the surrounding values
    :param operators: a list of values between +, -, * and / which are the operators to apply
    :param expression: list of math elements (numbers, operators) obtained after parsing
    :return: nothing, the expression list is modified by reference
    """

    i = 1
    while i < len(expression) - 1:

        if expression[i] in operators:
            operator = expression[i]
            op1 = expression[i - 1]
            op2 = expression[i + 1]

            # Apply the operation between the previous and following values
            if operator == '+':
                res = op1 + op2
            elif operator == '-':
                res = op1 - op2
            elif operator == '*':
                res = op1 * op2
            elif operator == '/':
                res = op1 / op2
            else:
                raise Exception("apply_operator() should only be called with valid operators!") 

            # Replace the 3 items (op1, operator, op2) with the operation result
            expression[i-1] = res
            del expression[i+1]
            del expression[i]

        else:
            i += 1      # Increment index
