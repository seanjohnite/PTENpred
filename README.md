# PTENpred

Making connections between genotype and phenotype has become more important than ever in recent years. Whole exome sequencing will be even more inexpensive as more and better techniques are developed, but in cancer, many of the genes responsible for transformation of normal cells are known. Current software for measuring the impact of non-synonymous single nucleotide polymorphisms is designed for analysis of all proteins. We present a protein impact predictor, PTENpred, that is trained on and meant to predict phenotypes related to a single important protein, PTEN. PTEN is implicated in most cancers and is also connected to germline disorders including Cowden syndrome and autism. Our predictor is designed for use by biologists, clinicians, and laymen interested in interpreting the possible effect of a novel PTEN mutation.

Here, PTENpred is implemented as a Python script. You need Python 2 to run it. If you're on a Mac or on Linux, you probably already have Python, but if you're on Windows, you should go to the [Python Software Foundation](https://www.python.org/) to find a distribution.

To run PTENpred, first clone and go into repository in your working directory with 

    git clone https://github.com/seanjohnite/PTENpred.git
    cd PTENpred

If you're going to be working with a lot of different Python projects, I would recommend setting up a virtual environment for PTENpred. This will isolate PTENpred packages from other packages you might have installed to limit conflicts. Start with

    pip install virtualenv
    virtualenv PTENpredenv

These commands will install the `virtualenv` package to your system and create an interpreter for PTENpred.

    source PTENpredenv/bin/activate

will "turn on" the Python version in your terminal. If everything worked, your command prompt should look like this:

    (PTENpredenv) computer-name:PTENpred user$

Now, you can install the packages required to run PTENpred just for the PTENpredenv virtual environment.

    pip install -r requirements.txt

will install all of the required packages for PTENpred.

`cd predict` into the folder where the scripts are, and run PTENpred with the following command syntax:

    ./classifyMutation.py [-c CAT_SPLIT] VARIANT
    
 `CAT_SPLIT` is a number that designates the way certain groups of mutations are categorized in the training data for each predictor. This is `2` by default, but can also be `22, 3,` or `4`.
 
 PTENpred will give the possible categories for a specified prediction as well as the predicted category.
