
with open('spambase', 'r') as original:
    with open('spambase_train', 'w+') as trainf:
        for i in range(3000):
            trainf.write(original.readline())
    with open('spambase_test', 'w+') as testf:
        for i in range(1601):
            testf.write(original.readline())
