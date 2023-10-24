# TOKTER
Code for the paper "Harnessing Test-Oriented Knowledge Graphs for Enhanced Test Function Recommendation"


## Usage

* run `build_kg` method in `build_knowledge_graph.py` to build the test-oriented knowledge graph using NetworkX tool.
* run `rec_test_functions` method in `recommendation.py` to recommend test functions.

## Instruction
* The `build_cache` method in `recommendation.py` is the method to create a path cache file before recommending test functions, which can increase the speed of recommendation.
* The `DomainConcept.txt` file is the list of domain concepts reviewed by domain experts.

## Citation