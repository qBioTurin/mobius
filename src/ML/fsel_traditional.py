import streamlit as st 
from itertools import product
from typing import List

import tools.ml_utils as mlu
from tools.ml_managers import FeatureSelectorManager
from tools.utils import HelpMessage


def traditional_feature_selection( 
        training_data: mlu.LabelledData, 
        filter_list: List[str], wrapper_list: List[str], embedded_list: List[ mlu.LearningAlgorithm ], 
        ub_nfeatures: int, underlying_est: List[ mlu.LearningAlgorithm ] ):
    
    fset_manager = st.session_state[ mlu.ML_SessionState.FEATURES.value ]
    
    scaled_training_data = training_data.scale_data()

    st.write(f"Doing non-parametric stuff")

    for fset_id, fset_list in FeatureSelectorManager.apply_non_parametric( scaled_training_data ).items():
        fset_manager.add_feature_set( fset_list, metadata = fset_id  )

    for filter_criterion in filter_list:
        st.write(f"**Filter FS**: {filter_criterion}")
        fset_manager.add_feature_set(
            FeatureSelectorManager.apply_filter( scaled_training_data, filter_criterion, ub_nfeatures ), 
            metadata = filter_criterion
        )

    for embedded_criterion in embedded_list:
        st.write(f"**Embedded FS**: {embedded_criterion}")
        fset_manager.add_feature_set(
            FeatureSelectorManager.apply_embedded( scaled_training_data, embedded_criterion, ub_nfeatures ),
            metadata = embedded_criterion
        )

    for wrap_criterion, u_clf in product( wrapper_list, underlying_est ):
        st.write(f"**Wrapper FS**: {wrap_criterion} x {u_clf}")
        fset_manager.add_feature_set(
            FeatureSelectorManager.apply_wrapper( scaled_training_data, wrap_criterion, u_clf ),
            metadata = f"{wrap_criterion}-{u_clf.value}"
        )

    mlu.update_feature_dataframe()


def traditional_fsel_UI():
    feature_found_df = st.session_state[mlu.ML_SessionState.FEATURE_FOUND.value ]

    with st.container(border=True):
        st.subheader("Feature sets")
        st.dataframe( feature_found_df, use_container_width = True )

        with st.form(f"form_choose_fsets", border=False):
            st.multiselect("Choose initial features: ", key="chosen_fsets", 
                options = feature_found_df.index.tolist(), 
                default = st.session_state.get( mlu.ML_SessionState.SELECTED_FEATURES.value )
                ) #st.session_state[ ML_SessionState.FEATURE_GROUPS.value ] )
            if st.form_submit_button("Set!"):
                st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ] = st.session_state.chosen_fsets 
                st.write(f"Current choice updated: {st.session_state.chosen_fsets}")
 

    with st.form("fs_form_1"):
        filter_fs = ["ANOVA", "MI"]
        wrapper_fs = ["RFE", "FFS"]


        c1, c2 = st.columns(2)
        with c1:
            st.pills("Filter FS", options=filter_fs, key="filter_methods", selection_mode="multi")
            st.pills("Embedded FS", key="embed_methods", selection_mode="multi", options = list( mlu.EmbeddedFeatureSelection ) )
            st.pills("Wrapper FS", options=wrapper_fs, key="wrapper_methods", selection_mode="multi")

        with c2:
            nf_tot = len( mlu.prepare_features( st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ], st.session_state ) ) 
            st.slider("Choose the number of selectable features for **filter and embedded selections**", min_value = 1, max_value=nf_tot, key="selectable_nf")
            st.pills(
                "Choose underlying algorithm for **wrapper selection only**", 
                key="wrapper_underlying", selection_mode="multi", 
                options = [ mlu.LearningAlgorithm.LINEAR_SVM, mlu.LearningAlgorithm.LOGISTIC_REGRESSION, mlu.LearningAlgorithm.DECISION_TREE ] )


        if st.form_submit_button("Run selection"):
            with st.status( f"Feature selection time!" ) as status:
                f_groups = st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ]
                initial_featurepool = mlu.prepare_features( f_groups, st.session_state )
                training_data = st.session_state[ mlu.ML_SessionState.DISCOVERY_SET.value ].select_features( initial_featurepool )

                traditional_feature_selection( 
                    training_data, 
                    st.session_state.filter_methods, 
                    st.session_state.wrapper_methods, 
                    st.session_state.embed_methods, 
                    st.session_state.selectable_nf, 
                    st.session_state.wrapper_underlying
                )

                status.update(label="Computation completed!", state="complete", expanded=False) 
                st.rerun()


st.title("Traditional Feature Selection")

HelpMessage.classic_fs()

mlu.ml_guard(st.session_state)

traditional_fsel_UI() 