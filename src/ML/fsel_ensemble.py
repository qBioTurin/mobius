import streamlit as st 
import tools.ml_utils as mlu
from tools.ml_managers import FeatureSelectorManager
from tools.utils import HelpMessage


def ensemble_fsel_UI():
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


    selected_features = mlu.prepare_features( st.session_state[ mlu.ML_SessionState.SELECTED_FEATURES.value ], st.session_state )
    max_nf = min( 30, len(selected_features ))

    with st.form("run_ensemble"):
        st.text_input("Select a name plz", key = "ensemble_name", max_chars=30, placeholder="")
        st.number_input("Number of bootstrap replicas", key="niter_ensemble", min_value=1, max_value=10000, value=50)
        st.slider("Select the range of features to consider", key="nf_ensemble", min_value = 1, max_value = max_nf, value = (1,10) )  #options=range(1,max_nf), value=(1,10), key="nf_ensemble")

        if st.form_submit_button("Run ensemble feature selection"):
            ensemble_name = st.session_state.ensemble_name.strip()
            if len( ensemble_name ) == 0:
                st.error("Plz choose a name!")
            elif st.session_state[ mlu.ML_SessionState.FEATURES.value ].get_ensemble( ensemble_name ) is not None:
                st.error(f"An ensemble named as **{ensemble_name}** already exists! Choose a different name!")
            else:
                with st.status("Running ensemble feature selection", expanded=True) as status:
                    min_nf, max_nf = st.session_state.nf_ensemble
                    
                    ensemble_set = FeatureSelectorManager.apply_ensemble(
                        st.session_state[ mlu.ML_SessionState.DISCOVERY_SET.value ],
                        selected_features,
                        min_nf, max_nf, 
                        st.session_state.niter_ensemble,
                        st_stream=st.write
                    )
                    
                    st.session_state[ mlu.ML_SessionState.FEATURES.value ].add_ensemble( ensemble_name, ensemble_set )
                    st.success(f"Ensemble feature set {ensemble_name} created with {len(ensemble_set)} features")

                    
                    status.update(label="Computation completed!", state="complete", expanded=False) 
                
 
st.title("Ensemble Feature Selection")

HelpMessage.ensemble_fs()

mlu.ml_guard(st.session_state)


ensemble_fsel_UI() 