
print '# example 1----------'

for i in range(0,5):
    print 'printing number: %s ' % i

print '# example 2---------'
namelist = ['kim','park','lee','kang']

for name in namelist:
    print 'pringing name: %s'  % name

print '# example 3 --------'

def myFuncPrint(keyword,numOfpring):
    for i in range(0,numOfpring):
        print 'The %s-th printing of %s' % (i+1,keyword)

myFuncPrint('Jaewookkang',10)



