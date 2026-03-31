import random

def generateExample(min_val, max_val):
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, a)  # ensure a >= b
    return f"{a}-{b}={a-b}\n"

def generateNDigitExample(numDigits):
    upperLimit = 10 ** (numDigits + 1) - 1
    lowerLimit = 10 ** numDigits
    return generateExample(lowerLimit, upperLimit)

def generate1DigitSimpleExamples():
    """
    1-digit subtraction (no borrowing, always true for 1-digit when a >= b)
    """
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, a)  # ensures a >= b
        return f"{a}-{b}={a-b}\n"

def generate1DigitBorrowExamples():
    """
    Included only to keep symmetry with addition generators.
    For 1-digit subtraction, borrowing never happens if a >= b,
    so this function is effectively unused.
    """
    while True:
        a = random.randint(0, 9)
        b = random.randint(0, a)
        # no valid 1-digit borrow case
        continue
        
def generate2DigitSimpleExamples():
    """
    2-digit subtraction WITHOUT borrowing:
    ones(a) >= ones(b) AND tens(a) >= tens(b)
    """
    while True:
        a = random.randint(10, 99)
        b = random.randint(10, a)  # ensure a >= b

        a1, a10 = a % 10, a // 10
        b1, b10 = b % 10, b // 10

        # no borrow in ones, no borrow in tens
        if a1 >= b1 and a10 >= b10:
            return f"{a}-{b}={a-b}\n"

def generate2DigitBorrowExamples():
    """
    2-digit subtraction WITH borrowing:
    ones(a) < ones(b)
    """
    while True:
        a = random.randint(10, 99)
        b = random.randint(10, a)  # ensure a >= b

        a1, a10 = a % 10, a // 10
        b1, b10 = b % 10, b // 10

        # borrow required from tens
        if a1 < b1:
            return f"{a}-{b}={a-b}\n"

def generateComplexExamples():
    """
    Larger-number subtraction, still enforcing a >= b
    """
    a = random.randint(1000, 9999)
    b = random.randint(10, a)
    return f"{a}-{b}={a-b}\n"