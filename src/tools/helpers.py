import enum 
import streamlit as st 
from typing import Union


class HelperEvaluation(enum.Enum):
    """
    Enum class to represent helper messages for the widgets in evaluation page
    """

    CLF_TO_EVALUATE = "Classifier(s) to be fitted on the training data and tested on the cross-validation data and the hold-out test data"
    IMBALANCED_TECHNIQUE = "Select one imbalanced technique to be applied (**nb. enabled for k-fold CV only**)"
    CV_TECHNIQUE = "Cross-validation technique to be applied. **K-fold CV** split the dataset into k folds, and each fold is used as a test set once. **Leave-one-out CV** uses all but one sample for training and the left-out sample for testing."
    N_REPS = "Number of times the k-fold cross-validation will be repeated"
    N_FOLDS = "Number of folds for k-fold cross-validation"


class HelperGeneticAlgorithm(enum.Enum):
    """
    Enum class to represent helper messages for the widgets in genetic algorithm page
    """

    INITIAL_FEATURES = "Initial features to be used to get the feature graph"
    P_THRESHOLD = "Select the p-value threshold such that edges with p-value > p_threshold will be cut from the graph"
    VERTEX_FILTERING = "Select the vertex filtering technique to be applied: "
    MIN_COMMUNITY_SIZE = "Select the minimum size of the communities to be detected"

    COMM_METHOD = "Select the community detection method to be selected "
    IMPORTANCE_METHOD = "Select the importance method to be selected "

    FITNESS_FUNCTION = "Fitness function guiding the mRMR-based FS: in **pure_graph**, relevance and redundancy are estimated based on vertex and edge weights; in **hybrid**, a classifier is trained in CV and the performance is used as relevance"
    POPULATION_SIZE = "Number of solutions in the population"
    RANDOM_SEED = "Random seed for reproducibility"
    DYNAMIC_REWEIGHTING = "Enable dynamic reweighting of the edges in the graph to weaken the edges proportionally to the weights of their incident vertices"
    SEEDED_POPULATION = "Enable the use of a seeded population: a fraction of the population is sampled based on the importance of the vertices in the graph"
    UB_N_GENERATIONS = "Upper bound on the number of generations to be run: the algorithm should converge before this number is reached, but who knows?"

    HYBRID_GUIDE = "Choose the ML algorithm to be used to guide the hybrid mRMR-based FS"
    GUIDE_METRIC = "Choose the metric to be maximized by the ML algorithm in the hybrid mRMR-based FS"


class HelpMessageFormatter:
    def __init__(self, 
            msg_intro: str,
            msg_prereq: str,
            msg_params: str = None,
            msg_output: str = None
        ):

        self.__msg = f"**{msg_intro}**\n\n"

        if msg_prereq:
            self.__msg += f"**Prerequisites**\n\n{msg_prereq}\n\n" 
    
        if msg_params:
            self.__msg += f"**Parameters**\n{msg_params}\n\n" 

        if msg_output:
            self.__msg += f"**Behavior and output**\n\n{msg_output}"
        
    @property
    def msg(self) -> str:
        return self.__msg
    
    def explanatory_msg(self):
        msg = self.__msg
        with st.expander("**What does this module do and how does it work?**", expanded=False):
            match msg:
                case HelpMessageFormatter():
                    st.info(msg.msg)
                case str():
                    st.info(msg)
        

class HelpMessage:

    @staticmethod
    def data_preparation():
        HelpMessageFormatter(
            msg_intro = "Initial data preparation for MOBIUS.",
            msg_prereq = "NA", #"You can remove metadata columns and filter out samples based on column value conditions (e.g., removing samples belonging to a specific cohort).  \nThe metadata table is matched against the omics tables, and only samples having all the omics layers are kept.\n\nAliases and colors should be assigned to omics layers.",
            msg_params = (
                "* **one** metadata file containing sample identifiers and demographic/clinical covariates\n"
                "* n >= 1 files containing n omics data matrices (e.g. gene expression, methylation, proteomics)."),
            msg_output = "Prepared data zip file" #"The final preprocessed data will be downloaded for the next steps."
        ).explanatory_msg()
        

    @staticmethod
    def network_build():
        HelpMessageFormatter(
            msg_intro = "Build correlation networks by stratifying samples using up to two covariates.",
            msg_prereq="📂 Data Preparation step",
            msg_params = (
"* Correlation coefficient used to compute edge weights -> edges have two weights: the value of the correlation coefficient and the associated p-value."
"* Omics interaction: select which omics to consider for intra- and inter-omics interactions."),
            msg_output = "Network set zip file containing the correlation networks and the data used to build them."
        ).explanatory_msg()
    

    @staticmethod
    def init_task():
        HelpMessageFormatter(
            msg_intro = "Initialize a binary classification task.",
            msg_prereq="📂 Data Preparation step",
            msg_params = "* Target covariate: the covariate to be predicted\n" \
                         "* Positive class: the value(s) of the target covariate to be considered as the positive class\n" \
                         "* Negative class: the value(s) of the target covariate to be considered as the negative class\n" \
                         "* Test set size (%): percentage of samples to be reserved as test set\n" \
                         "* Sample stratification covariates: an optional list of covariates to be used for balancing training/test set split\n",
            msg_output = "Preprocessed data zip file with the task configuration."
        ).explanatory_msg()


    @staticmethod
    def load_zip_nets():
        HelpMessageFormatter(
            msg_intro = "Load the networks and set up the analysis parameters.",
            msg_prereq="📂 Data Preparation step -> 🌐 Network Preparation",
            msg_params = "You can select between single-graph mode and multi-graph mode:\n* In single-graph mode, you can select one network at a time and explore the strata defined by the internal stratification covariate.\n* In multi-graph mode, you can select multiple networks at once to be analyzed. The 'WholeData' network will be considered, and the internal stratification covariate is ignored in this mode.",
            msg_output = "A zip file containing tables and plots visualized in ."
        ).explanatory_msg()
        

    @staticmethod
    def analyze_nets():
        HelpMessageFormatter(
            msg_intro = "Look for important features according to centrality measures, communities organization, and most correlated feature pairs",
            msg_prereq="📂 Data Preparation step -> 🌐 Network Preparation -> ⚜️ Load Zipped Networks",
            msg_output = "The results of the analysis will be saved to a specified output location."
        ).explanatory_msg()

        
    @staticmethod
    def viz_nets():
        HelpMessageFormatter(
            msg_intro = "Interactive network visualization",
            msg_prereq="📂 Data Preparation step -> 🌐 Network Preparation -> ⚜️ Load Zipped Networks",
            msg_output="").explanatory_msg()


    @staticmethod
    def prune_nets():
# This module prunes the correlation networks to remove weak or irrelevant edges.
# """
        HelpMessageFormatter(
            msg_intro = "Double-stratified network pruning based on internal covariate",# = "Visualize the correlation networks to provide insights into the relationships between variables.", 
            msg_prereq="📂 Data Preparation step -> 🌐 Network Preparation",
            msg_params="", 
            msg_output=""
        ).explanatory_msg()

    @staticmethod
    def overlap_nets():
# This module identifies overlapping edges between different correlation networks.
# """
        HelpMessageFormatter(
            msg_intro = "Prune edges in double-stratified networks based on the internal covariate", #"Visualize the correlation networks to provide insights into the relationships between variables.", 
            msg_prereq="📂 Data Preparation step -> 🌐 Network Preparation -> ⚜️ Load Zipped Networks",
            msg_params="", 
            msg_output="").explanatory_msg()


    @staticmethod
    def load_zip_task():
# This module loads the preprocessed data from a zip file and reconstructs the correlation networks.
# """
        HelpMessageFormatter(
            msg_intro = "Load a task into session and enable ML analysis modules",# = "Visualize the correlation networks to provide insights into the relationships between variables.", 
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation",
            msg_params="", 
            msg_output="").explanatory_msg()
    
    @staticmethod
    def handle_features():
        """ Manage and visualize the available feature sets identified so far. The user can manually add new feature sets and remove the ones that are not of interest. From here, feature sets identified from ensemble FS can be visualized, together with saved generations from net- based FS.  THe user can apply similarity measures to compare feature sets and identify overlapping ones. """

        HelpMessageFormatter(
            msg_intro=(
                "🗂️ Manage Feature Sets: Visualize, compare, and curate the feature sets "
                "identified in previous steps."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task",
            msg_params=(
                "- **Feature set selection**: Choose which feature set(s) to view or modify.\n"
                "- **Add/Remove feature sets**: Manually include new sets or discard irrelevant ones.\n"
                "- **Comparison**: Apply similarity measures to detect overlaps and relationships between sets."
            ),
            msg_output=(
                "As output, users can manage the feature sets available for downstream modules, "
                "visualize differences or overlaps, and curate sets for further evaluation or analysis."
            )
        ).explanatory_msg()

    @staticmethod
    def eda():
        """ Exploratory data analysis (EDA) to get descriptive statistics, plots like PCA, tSNE and stuff, and distributions to better understand your dataset
        The user can select which feature set to analyze from the available ones."""


        HelpMessageFormatter(
            msg_intro=(
                "📈 Exploratory Data Analysis (EDA): Explore your dataset through "
                "descriptive statistics, dimensionality reduction, and visualization."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task" ,
            msg_params=(
                "- **Feature set selection**: Choose which feature set to analyze from those "
                "available in the MongoDB database.\n"
                "- **Dimensionality reduction methods**: PCA, t-SNE, and other techniques "
                "for exploring feature space structure.\n"
                # "- **Visualization options**: Distribution plots, scatter plots, and "
                # "summary statistics."
            ),
            msg_output=(
                "As output, users obtain interactive plots and descriptive statistics that "
                "help characterize the selected feature set. These visualizations allow a "
                "better understanding of the data structure, sample distributions, and "
                "feature variability."
            )
        ).explanatory_msg()
# This module performs exploratory data analysis (EDA) on the correlation networks.
# """
    
    @staticmethod
    def classic_fs():
        """ Classic feature selection (i.e. filter, wrapper, embedded) from scikit-learn library. 
        Among filters: ANOVA and multual information. 
        Wrapper: RFE and FFS
        Embedded: st.pills("Embedded FS", key="embed_methods", options = [ mlu.LearningAlgorithm.LINEAR_SVM, mlu.LearningAlgorithm.LOGISTIC_REGRESSION, mlu.LearningAlgorithm.DECISION_TREE, mlu.LearningAlgorithm.LINEAR_DA ], selection_mode="multi" )
        Non-parametric stuff: AIC and BIC with logistic regression 
        """

        HelpMessageFormatter(
            msg_intro=(
                "📊 Classic Feature Selection: Apply filter, wrapper, and embedded methods "
                "provided by the scikit-learn library."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task",
            msg_params=(
                "- **Input feature set**: Choose the feature set from MongoDB that will serve as the starting point for the feature selection procedure."
                "- **Filter methods**: ANOVA F-test, Mutual Information, AIC, BIC "
                "(information-theoretic, model-dependent filters).\n"
                "- **Wrapper methods**: Recursive Feature Elimination (RFE), Forward Feature Selection (FFS).\n"
                "- **Embedded methods**: Linear SVM, Logistic Regression, Decision Tree, "
                "Linear Discriminant Analysis."
            ),
            msg_output=(
                "As output, the identified feature sets are saved in the MongoDB database. "
                "These sets can be accessed from other modules to be further evaluated, "
                "explored, and compared, ensuring smooth integration within the workflow."
            )
        ).explanatory_msg()

    
    @staticmethod  
    def ensemble_fs():
        """ Ensemble feature selection by combining filter and embedded methods in a majority voting schema. 
        Multiple feature selections are applied simultaneously, and features are ranked based on the number of times they are selected.
        This process is repeated multiple times exploiting repeated cross-validation, and a final set of features is obtained by selecting the top-k ranked features. 
        This process is repeated using an incremental number of features, given the minimum and maximum number of features to be selected..
        As output, a set of feature sets is provided, each one corresponding to a different number of selected features. 
        This can be evaluated from the model evaluation module for identifying a trade-off between feature set complexity and prediction performances. """


        HelpMessageFormatter(
            msg_intro=(
                "🧮 Ensemble Feature Selection: Combine filter and embedded methods through "
                "a majority voting scheme to rank and select features."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task",
            msg_params=(
                "- **Input feature set**: Choose the feature set from MongoDB that will serve as the starting point for the feature selection procedure."
                # "- **Feature selection methods**: Choose multiple filter and embedded techniques "
                # "to be combined in a majority voting schema.\n"
                "- **Cross-validation**: Specify the number of repetition and of folds for CV\n"
                "- **Feature range**: Define the minimum and maximum number of features to be selected.\n"
                # "- **Incremental selection**: Generate feature sets with increasing numbers of features "
                # "within the specified range."
            ),
            msg_output=(
                "As output, users obtain a series of feature sets, each corresponding to a different "
                "number of selected features. These sets can be passed to the Model Training module "
                "to evaluate trade-offs between feature set complexity and predictive performance."
            )
        ).explanatory_msg()

    
    @staticmethod
    def net_fs():
        """ Network-based feature selection techniques to identify sets of highly relevant and low-redundant features by leveraging a multi-objective evolutionary algorithm running on the feature graph.
        Two modes are provided: pure-graph mode, where relevance and redundancy are estimated based on vertex and edge weights; hybrid mode, where a classifier is trained in CV and the performance is used as relevance.
        For pure-graph mode, the user can choose among different feature importance scores, whereas in hybrid mode the user can choose the ML algorithm and the performance metric to be optimized.
        In any case, the user has to choose the community detection algorithm to be used to identify groups of related features in the graph. As output, the user can visualize the relevance/redundancy otimization process and 
         can explore the generations explored by the MOEA and select the feature sets to be evaluated in the next step. As default, the last generation is shown.
        """

        HelpMessageFormatter(
            msg_intro=(
                "🌐 Network-based Feature Selection: Identify feature sets that maximize relevance "
                "and minimize redundancy using a feature correlation graph and a multi-objective "
                "evolutionary algorithm."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task",
            msg_params=(
                "- **Input feature set**: Choose the feature set from MongoDB that will serve as the starting point for the feature selection procedure."
                "- **Mode selection**:\n"
                "  • *Pure-graph mode*: relevance and redundancy estimated from vertex and edge weights.\n"
                "  • *Hybrid mode*: train a classifier with cross-validation and use performance as relevance.\n"
                "- **Feature importance (pure-graph mode)**: choose among available scoring functions.\n"
                "- **Classifier (hybrid mode)**: select ML algorithm to be trained.\n"
                "- **Performance metric (hybrid mode)**: define which metric to optimize.\n"
                "- **Community detection algorithm**: required to identify groups of related features.\n"
                "  Community structure is exploited to apply a *divide-et-impera* approach for redundancy estimation."
            ),
            msg_output=(
                "As output, users can visualize the optimization process that balances feature relevance "
                "and redundancy. The exploration history of the MOEA is available, allowing users to browse "
                "through the generations produced during the search. From these generations, users can select "
                "the feature sets to be carried forward for evaluation in the next step. By default, the final "
                "generation is displayed."
            )
        ).explanatory_msg()

    
    @staticmethod
    def model_training():
        """ Allows users to select n feature sets and evaluate them using m ML models. Evaluation can be done either using k-fold or leave-one-out cross-validation, and results are also reported on the hold-out test set.
        As outptut, performance metrics are reported for each feature set / ML model combination and ROC AUC and PR AUC curves are plotted.
        """
# This module trains machine learning models on the selected features from the correlation networks.
# """

        HelpMessageFormatter(
            msg_intro=(
                "🤖 Model Training: Train and evaluate multiple machine learning models "
                "on user-selected feature sets."
            ),
            msg_prereq="📂 Data Preparation step -> 🏗️ Task Preparation -> ✨ Load Zipped Task",
            msg_params=(
                "- **Feature set selection**: Choose one or more feature sets to evaluate.\n"
                "- **ML model selection**: Specify which machine learning models to train.\n"
                "- **Validation strategy**: Select either k-fold or leave-one-out cross-validation.\n"
                "- **Hold-out test set**: Optionally use a separate test set for final evaluation."
            ),
            msg_output=(
                "- Performance metrics (e.g., AUC, accuracy, precision, recall, F1) for each "
                "feature set / model combination.\n"
                "- ROC AUC and PR AUC curves for visual comparison across models."
            )
        ).explanatory_msg()