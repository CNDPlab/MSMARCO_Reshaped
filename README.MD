# Machine Reading Comprehension with MS-MARCO v2.1 (On progress)
## System details 

* Data process: 
    feature: word & char & elmo(TODO)  
    label: boundary span(rouge score > 0.8) & distribution(TODO)
* Model :  
    SNET  
    VNET(TODO)
    
## Usage

* Preprocess:   
python preprocess.py
  
* train:
python main.py train --batch_size=32

