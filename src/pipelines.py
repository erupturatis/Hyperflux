from src.test1.training import run
from src.test1.validate_nn import validate
from src.test2.training import run_conv2

#For FC 
def pipeline_test1():
    run()
    
def pipeline_validate():
    validate()
    
#For Conv2 
def pipeline_test2():
    run_conv2()
    
