Search.setIndex({docnames:["code/create_trainable","code/data","code/data.CarbonIntensity","code/data.Weather","code/data.Workload","code/dcrl_env","code/dcrl_eplus_env","code/envs","code/evaluate_model","code/folder_structure","code/index","code/maddpg","code/modules","code/train","code/train_a2c","code/train_maddpg","code/train_ppo","code/utils","contribution_guidelines","gettingstarted","index","installation/custom","installation/energyPlus","installation/index","installation/installationtips","overview/actions","overview/agents","overview/environment","overview/index","overview/observations","overview/reward_function","references","train_evaluate/evaluate","train_evaluate/index","train_evaluate/monitor_results","usage/index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,"sphinxcontrib.bibtex":9,sphinx:56},filenames:["code/create_trainable.rst","code/data.rst","code/data.CarbonIntensity.rst","code/data.Weather.rst","code/data.Workload.rst","code/dcrl_env.rst","code/dcrl_eplus_env.rst","code/envs.rst","code/evaluate_model.rst","code/folder_structure.rst","code/index.rst","code/maddpg.rst","code/modules.rst","code/train.rst","code/train_a2c.rst","code/train_maddpg.rst","code/train_ppo.rst","code/utils.rst","contribution_guidelines.rst","gettingstarted.rst","index.rst","installation/custom.rst","installation/energyPlus.rst","installation/index.rst","installation/installationtips.rst","overview/actions.rst","overview/agents.rst","overview/environment.rst","overview/index.rst","overview/observations.rst","overview/reward_function.rst","references.rst","train_evaluate/evaluate.rst","train_evaluate/index.rst","train_evaluate/monitor_results.rst","usage/index.rst"],objects:{"":[[0,0,0,"-","create_trainable"],[1,0,0,"-","data"],[5,0,0,"-","dcrl_env"],[6,0,0,"-","dcrl_eplus_env"],[7,0,0,"-","envs"],[8,0,0,"-","evaluate_model"],[11,0,0,"-","maddpg"],[13,0,0,"-","train"],[14,0,0,"-","train_a2c"],[15,0,0,"-","train_maddpg"],[16,0,0,"-","train_ppo"],[17,0,0,"-","utils"]],"dcrl_env.DCRL":[[5,3,1,"","calculate_reward"],[5,3,1,"","reset"],[5,3,1,"","step"]],"dcrl_env.EnvConfig":[[5,4,1,"","DEFAULT_CONFIG"]],"dcrl_eplus_env.DCRLeplus":[[6,3,1,"","calculate_reward"],[6,3,1,"","reset"],[6,3,1,"","step"]],"envs.bat_env_fwd_view":[[7,2,1,"","BatteryEnvFwd"]],"envs.bat_env_fwd_view.BatteryEnvFwd":[[7,3,1,"","CO2_footprint"],[7,3,1,"","charging_rate_modifier"],[7,3,1,"","discharging_rate_modifier"],[7,3,1,"","reset"],[7,3,1,"","set_dcload"],[7,3,1,"","step"],[7,3,1,"","update_ci"],[7,3,1,"","update_state"]],"envs.battery_model":[[7,2,1,"","Battery"],[7,2,1,"","Battery2"],[7,1,1,"","apply_battery"],[7,1,1,"","calculate_247_battery_capacity"],[7,1,1,"","calculate_247_battery_capacity_b1_sim"],[7,1,1,"","calculate_247_battery_capacity_b2_sim"],[7,1,1,"","sim_battery_247"]],"envs.battery_model.Battery":[[7,4,1,"","capacity"],[7,3,1,"","charge"],[7,4,1,"","current_load"],[7,3,1,"","discharge"],[7,3,1,"","find_and_init_capacity"],[7,3,1,"","is_full"]],"envs.battery_model.Battery2":[[7,4,1,"","c_lim"],[7,3,1,"","calc_max_charge"],[7,3,1,"","calc_max_discharge"],[7,4,1,"","capacity"],[7,3,1,"","charge"],[7,4,1,"","current_load"],[7,4,1,"","d_lim"],[7,3,1,"","discharge"],[7,4,1,"","eff_c"],[7,4,1,"","eff_d"],[7,3,1,"","find_and_init_capacity"],[7,3,1,"","is_full"],[7,4,1,"","lower_lim_u"],[7,4,1,"","lower_lim_v"],[7,4,1,"","upper_lim_u"],[7,4,1,"","upper_lim_v"]],"envs.carbon_ls":[[7,2,1,"","CarbonLoadEnv"]],"envs.carbon_ls.CarbonLoadEnv":[[7,3,1,"","reset"],[7,3,1,"","step"],[7,3,1,"","update_workload"]],"envs.datacenter":[[7,2,1,"","CPU"],[7,2,1,"","DataCenter_ITModel"],[7,2,1,"","Rack"],[7,1,1,"","calculate_HVAC_power"],[7,1,1,"","calculate_avg_CRAC_return_temp"]],"envs.datacenter.CPU":[[7,3,1,"","compute_instantaneous_cpu_pwr"],[7,3,1,"","compute_instantaneous_fan_pwr"],[7,3,1,"","cpu_curve1"],[7,3,1,"","itfan_curve2"]],"envs.datacenter.DataCenter_ITModel":[[7,3,1,"","compute_datacenter_IT_load_outlet_temp"],[7,3,1,"","total_datacenter_full_load"]],"envs.datacenter.Rack":[[7,3,1,"","compute_instantaneous_pwr"],[7,3,1,"","get_average_rack_fan_v"],[7,3,1,"","get_current_rack_load"]],"envs.dc_gym":[[7,2,1,"","dc_gymenv"],[7,2,1,"","dc_gymenv_standalone"]],"envs.dc_gym.dc_gymenv":[[7,3,1,"","NormalizeObservation"],[7,3,1,"","get_obs"],[7,3,1,"","normalize"],[7,3,1,"","reset"],[7,3,1,"","set_ambient_temp"],[7,3,1,"","set_bat_SoC"],[7,3,1,"","set_shifted_wklds"],[7,3,1,"","step"]],"maddpg.MADDPGConfigStable":[[11,3,1,"","validate"]],"maddpg.MADDPGStable":[[11,3,1,"","get_default_config"],[11,3,1,"","get_default_policy_class"]],"utils.base_agents":[[17,2,1,"","BaseBatteryAgent"],[17,2,1,"","BaseHVACAgent"],[17,2,1,"","BaseLoadShiftingAgent"]],"utils.base_agents.BaseBatteryAgent":[[17,3,1,"","do_nothing_action"]],"utils.base_agents.BaseHVACAgent":[[17,3,1,"","do_nothing_action"]],"utils.base_agents.BaseLoadShiftingAgent":[[17,3,1,"","do_nothing_action"]],"utils.dc_config":[[17,5,1,"","CT_REFRENCE_AIR_FLOW_RATE"]],"utils.dc_config_reader":[[17,5,1,"","CT_REFRENCE_AIR_FLOW_RATE"]],"utils.helper_methods":[[17,1,1,"","f2c"],[17,2,1,"","pyeplus_callback"]],"utils.helper_methods.pyeplus_callback":[[17,3,1,"","on_episode_end"],[17,3,1,"","on_episode_start"],[17,3,1,"","on_episode_step"]],"utils.make_envs":[[17,1,1,"","make_bat_fwd_env"],[17,1,1,"","make_dc_env"],[17,1,1,"","make_ls_env"]],"utils.make_envs_pyenv":[[17,1,1,"","make_bat_fwd_env"],[17,1,1,"","make_dc_pyeplus_env"],[17,1,1,"","make_ls_env"]],"utils.managers":[[17,2,1,"","CI_Manager"],[17,2,1,"","CoherentNoise"],[17,2,1,"","Time_Manager"],[17,2,1,"","Weather_Manager"],[17,2,1,"","Workload_Manager"],[17,1,1,"","normalize"],[17,1,1,"","sc_obs"],[17,1,1,"","standarize"]],"utils.managers.CI_Manager":[[17,3,1,"","get_total_ci"],[17,3,1,"","reset"],[17,3,1,"","step"]],"utils.managers.CoherentNoise":[[17,3,1,"","generate"]],"utils.managers.Time_Manager":[[17,3,1,"","isterminal"],[17,3,1,"","reset"],[17,3,1,"","step"]],"utils.managers.Weather_Manager":[[17,3,1,"","get_total_weather"],[17,3,1,"","reset"],[17,3,1,"","step"]],"utils.managers.Workload_Manager":[[17,3,1,"","get_total_wkl"],[17,3,1,"","reset"],[17,3,1,"","step"]],"utils.reward_creator":[[17,1,1,"","custom_agent_reward"],[17,1,1,"","default_bat_reward"],[17,1,1,"","default_dc_reward"],[17,1,1,"","default_ls_reward"],[17,1,1,"","energy_PUE_reward"],[17,1,1,"","energy_efficiency_reward"],[17,1,1,"","get_reward_method"],[17,1,1,"","renewable_energy_reward"],[17,1,1,"","temperature_efficiency_reward"],[17,1,1,"","tou_reward"]],"utils.rllib_callbacks":[[17,2,1,"","CustomCallbacks"]],"utils.rllib_callbacks.CustomCallbacks":[[17,3,1,"","on_episode_end"],[17,3,1,"","on_episode_start"],[17,3,1,"","on_episode_step"]],"utils.utils_cf":[[17,1,1,"","get_energy_variables"],[17,1,1,"","get_init_day"],[17,1,1,"","obtain_paths"]],create_trainable:[[0,1,1,"","create_wrapped_trainable"]],data:[[2,0,0,"-","CarbonIntensity"],[3,0,0,"-","Weather"],[4,0,0,"-","Workload"]],dcrl_env:[[5,2,1,"","DCRL"],[5,2,1,"","EnvConfig"]],dcrl_eplus_env:[[6,2,1,"","DCRLeplus"]],envs:[[7,0,0,"-","bat_env_fwd_view"],[7,0,0,"-","battery_model"],[7,0,0,"-","carbon_ls"],[7,0,0,"-","datacenter"],[7,0,0,"-","dc_gym"]],evaluate_model:[[8,1,1,"","run"]],maddpg:[[11,2,1,"","MADDPGConfigStable"],[11,2,1,"","MADDPGStable"],[11,1,1,"","before_learn_on_batch"]],train:[[13,1,1,"","train"]],utils:[[17,0,0,"-","base_agents"],[17,0,0,"-","dc_config"],[17,0,0,"-","dc_config_reader"],[17,0,0,"-","helper_methods"],[17,0,0,"-","make_envs"],[17,0,0,"-","make_envs_pyenv"],[17,0,0,"-","managers"],[17,0,0,"-","reward_creator"],[17,0,0,"-","rllib_callbacks"],[17,0,0,"-","utils_cf"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:data"},terms:{"0":[5,7,17,19,25,27,30,34,35],"00":27,"00005663":27,"00009438":27,"001":17,"00e":[19,35],"01":[7,17],"02":[27,35],"02275":31,"025":17,"04":[7,21],"05":[27,35],"06":[19,35],"07":35,"09533":31,"1":[5,7,17,25,30,31,34,35],"10":[17,21,22,27,30,31],"100":27,"1000":27,"10000":7,"1006":27,"11":30,"110":27,"110603":17,"1145":31,"12":[19,30,35],"120":27,"123":7,"12th":17,"13":30,"130":27,"14":30,"1403":17,"15":30,"150":27,"16":30,"160":27,"160000":30,"17":30,"170":27,"1706":31,"18":[27,30,31],"180":27,"19":30,"190":27,"2":[5,17,23,25,30,31,34,35],"20":[21,30],"200":17,"2000":31,"2010":17,"2011":31,"2014":17,"2018":[17,31],"2020":31,"2021":[17,31],"2022":35,"20been":17,"20design":17,"20ha":17,"20inch":17,"20of":17,"20tower":17,"20water":17,"21":30,"22":[27,30],"220":27,"225":27,"23":[17,30],"231":17,"24":[17,19,27,30,35],"2443":30,"24gb":22,"25":30,"264":17,"27":[27,30],"280":27,"29":30,"3":[7,17,21,25,30,31,34,35],"30":[17,30],"300":27,"31":[30,35],"319":31,"323":31,"3265723":31,"3265742":31,"33":30,"3486611":31,"3488729":31,"365":35,"380899":30,"3906":17,"4":[7,17,22,25,27,30,31],"40000":30,"41":30,"42":[17,31],"420":27,"49":31,"5":[17,25,27,30,31,35],"50":30,"500":30,"530":27,"56":31,"6":[17,25,27,30],"60":[27,30],"610":30,"63":30,"650":27,"660":27,"670":27,"69":30,"6912":[19,35],"6sigma":27,"7":[25,27,30],"70":27,"760":30,"8":[5,19,25,30],"80":27,"81":30,"82":30,"8315":[17,27],"8315243039169418":17,"870":27,"8th":31,"9":[17,25,30],"99":[19,35],"9th":31,"bj\u00f6rn":17,"boolean":[7,17],"case":[11,30,31,34],"class":[5,6,7,10,11,17,34],"default":[11,17,19,22,27,30,35],"do":[17,24,25],"final":34,"float":[5,6,7,17,29,30],"function":[7,17,18,20,25,28,32,34,35],"g\u00f3mez":31,"import":[22,34],"int":[5,6,7,17,29,30],"jim\u00e9nez":31,"new":[18,19,30,31,34],"return":[0,5,6,7,11,17,27,30],"true":[7,17,35],A:[7,11,17,19,22,30],By:18,For:[19,35],IT:[7,17,27,30,35],ITE:17,If:[0,17,21,22,30,35],In:[19,30,31,32,34],Is:[30,31],It:[17,20,22,26,27,30],No:25,The:[0,7,10,17,18,19,20,22,23,25,27,29,30,32,34,35],These:18,To:[22,30,34],_avgci:[5,9,19,35],_lib:[7,17],a_t:7,ab:30,abbeel:31,abil:[20,23],about:7,abov:[32,35],absolut:30,accept:18,access:17,acm:31,across:17,action:[5,6,7,17,28,30,35],action_definit:7,action_dict:[5,6],action_id:7,action_map:7,action_spac:7,action_vari:7,actions_are_logit:5,activ:[21,30],actor:31,ad:[17,30,34],add:[11,17,18,30,33],add_ci:17,add_cpu_usag:[7,17],addit:[7,17,30,32],adher:18,adjust:[27,32],administr:35,advanc:23,advantag:35,advis:18,after:[30,32,34],agent:[5,6,7,11,17,20,25,27,28,29,30,31,32,34],agent_bat:[5,25,29,30,35],agent_dc:[5,17,25,29,30,35],agent_l:[5,25,29,30,35],agents_bat:19,agents_dc:19,agents_l:19,agnprz:22,aim:30,air:[7,27],al:17,alarm:[17,29],alejandro:31,alg:0,algo_class:11,algorithm:[0,11,13,17,19,20,30,34],algorithm_config:[11,13],algorithmconfig:[11,13],ali:31,alibaba:[31,35],alibaba_cpu_data_hourly_1:[5,9,17,19,35],alibaba_cpu_data_hourly_2:9,align:18,all:[7,30,31,34],allow:[20,23,27,30,34],along:[27,34],alreadi:30,also:[22,34],ambient_temp:[7,29],amount:[7,17,30,34],amout:7,an:[0,7,11,17,18,22,30,31,34],analysi:17,analyz:32,angel:9,ani:[5,6,11,18],anoth:30,antonio:31,anwar:31,apex:11,apply_batteri:7,approach:[7,30,35],appropri:34,apsi:31,ar:[17,18,19,22,24,30,34,35],architectur:20,arg:30,argument:17,arxiv:31,ashra:31,asia:31,assert:30,assign:[7,19,25,29],associ:31,attribut:[7,30],auxiliari:[17,20],avail:[10,27,30],averag:[7,34,35],average_battery_soc:34,avg_crac_return_temp:7,aviv:31,awar:20,az:35,azp:17,azps_ng_:9,b:[7,31],base:[5,6,7,11,17,26,30,35],base_ag:[9,12],base_env:[17,34],basebatteryag:17,baseenv:17,basehvacag:17,baseloadshiftingag:17,bat:[19,30],bat_a_t:30,bat_act:30,bat_avg_ci:30,bat_co2_footprint:30,bat_dcload_max:30,bat_dcload_min:30,bat_env_fwd_view:[9,12],bat_max_bat_cap:30,bat_reward:[5,6,30],bat_soc:[7,30],bat_total_energy_with_battery_kwh:30,bat_total_energy_without_battery_kwh:30,batch:[19,35],bateri:17,batsoc:29,batt:7,batteri:[5,6,7,17,19,20,25,26,29,30,34,35],battery2:7,battery_capac:7,battery_env_fwd:17,battery_model:[9,12],batteryenvfwd:7,becaus:30,befor:[18,19,30,35],before_learn_on_batch:11,begin:17,behavior:17,being:17,below:30,best:18,between:[20,30],blog:17,bool:[7,17],bound:[17,30],box:7,bpat:17,branch:18,breen:17,budget:25,bug:18,build:[17,20,29,30,31],built:20,c:[30,31],c_air:27,c_lim:7,cabinet:[17,27],calc_max_charg:7,calc_max_discharg:7,calcul:[5,6,7,17,27,30],calculate_247_battery_capac:7,calculate_247_battery_capacity_b1_sim:7,calculate_247_battery_capacity_b2_sim:7,calculate_avg_crac_return_temp:7,calculate_hvac_pow:7,calculate_reward:[5,6],call:[11,17,19,34],callabl:[11,17],callback:[17,34],campoi:31,can:[17,18,19,20,22,24,27,30,32,34,35],capac:[7,17,19,27,30],carbon:[1,2,7,17,19,20,26,27,29,30,34],carbon_l:[9,12],carbon_sustain:22,carbonintens:[1,9,12,35],carbonloadenv:[7,17],cd:[21,22],cdot:30,celsiu:17,center:[5,6,7,17,20,22,23,26,27,30,34,35],centr:17,central:17,centralis:35,cfd:27,chai:31,challeng:31,chang:[7,18,21,25,30],character:31,charg:[7,17,25,26,29,30],charging_r:[7,17],charging_rate_modifi:7,checkpoint:32,cheng:31,chiller:[7,27],chiller_cop:27,chip:17,choic:[23,35],choos:30,christian:31,ci:[7,27,29,30,35],ci_manag:17,ci_n:7,cintensity_fil:[5,35],circumst:18,citi:31,classmethod:11,clip:35,clip_param:35,clone:[21,22],close:30,cluster:17,co2:[30,34],co2_footprint:7,co2_footprint_mean:34,co:[23,31],code:[18,30],codebas:18,coeffici:[27,35],coher:17,coherentnois:17,collabor:[19,27,29],collect:[34,35],com:[17,21,22,24],combin:30,come:[17,30],command:[19,22,34,35],commit:18,common:17,competit:31,complet:[18,19],comprehens:18,compressor:[7,30],compressor_load:7,comput:[17,30,31,34],compute_datacenter_it_load_outlet_temp:7,compute_instantaneous_cpu_pwr:7,compute_instantaneous_fan_pwr:7,compute_instantaneous_pwr:7,concis:18,conda:21,condit:26,confer:[17,31],config:[11,13,30],configur:[19,23,30],connect:20,consid:30,consist:[30,35],constant:[27,30],consum:[17,30,34],consumpt:[7,17,20,27,29,30],contain:[7,17,23,30,35],content:12,continu:34,contribut:[20,35],control:[7,17,19,20,22,25,29,31,34,35],convert:17,cool:[7,17,19,20,25,26,27,29,30,35],cooper:[20,31],core:7,correct:[17,18],correspond:[7,17,18],cos_day_of_year:29,cos_time_hour:29,cosin:[17,29],cost:[20,30],could:[22,27],cpp:24,cpu:[7,17,27,29],cpu_config:7,cpu_config_list:7,cpu_curve1:7,cpu_load:7,cpu_pow:7,cpu_power_ratio_lb:27,cpu_power_ratio_ub:27,cpu_usag:29,cpus_per_rack:27,crac:[7,27,30],crac_cooling_load:7,crac_fan_load:7,crac_fan_ref_p:27,crac_refrence_air_flow_rate_pu:27,crac_setpoint:7,crac_supply_air_flow_rate_pu:27,crawlei:31,creat:[0,17,18,20,21,22,32,34],create_train:[9,12],create_wrapped_train:0,critic:[31,35],csv:[5,9,17,19,32,35],ct:17,ct_cooling_load:17,ct_fan_pwr:7,ct_fan_ref_p:27,ct_refrence_air_flow_r:[17,27],current:[7,17,21,25,29,30],current_dai:17,current_hour:17,current_load:7,current_pric:30,current_temperatur:[17,30],curti:31,curv:7,custom:[17,20,23,33],custom_agent_reward:[17,30],custom_metr:34,custom_reward:30,custom_sinergym:9,customcallback:[17,34],customiz:27,cython:24,d_lim:7,dai:[17,29,30],daili:[7,17],dashboard:34,data:[5,6,7,9,12,17,19,20,22,23,26,27,29,30,32,34],data_center_configur:27,data_center_full_load:7,data_center_total_ite_load:17,data_processor:9,databas:35,datacent:[9,12,30,31],datacenter_env:17,datacenter_itmodel:7,dataset:[30,35],davi:9,day_of_the_year:30,day_workload:[7,30],days_per_episod:17,dc:[7,9,17,19,23,25,29,30,35],dc_config:[7,9,12,27],dc_config_read:[9,12],dc_cpu_workload_perc:30,dc_crac_setpoint:30,dc_crac_setpoint_delta:30,dc_energy_lb_kw:30,dc_energy_ub_kw:30,dc_gym:[9,12,17],dc_gymenv:[7,17],dc_gymenv_standalon:7,dc_hvac_total_power_kw:30,dc_int_temperatur:30,dc_ite_total_power_kw:30,dc_itmodel_config:7,dc_load:7,dc_reward:[5,6,30],dc_total_power_kw:30,dcload_max:17,dcload_min:17,dcrl:[5,18,22,23,26,27,30,33,34,35],dcrl_env:[9,12,19,30],dcrl_eplus_env:[9,12,30],dcrleplu:[6,23,30,35],dcx:27,de:31,decentralis:35,decid:26,decis:[26,29],declar:30,decompos:35,def:[30,34],default_bat_reward:[5,17,30],default_config:5,default_dc_reward:[5,17,30],default_ls_reward:[5,17,30],default_server_power_characterist:27,defaultcallback:17,defin:[5,6,17,20,30],definit:30,degre:20,delta:[7,17,30],demand:[26,29],deni:31,densiti:27,deped:7,depend:[22,26,30],depth:20,describ:[25,29,35],descript:[18,19,25,27,29,30,35],design:[20,23],desir:[30,34],desired_std_dev:17,detail:[18,19,23,35],determin:32,devic:24,df_dc_pow:7,df_ren:7,dict:[5,6,7,13,17,30],dictionari:[5,6,7,17,30,34],differ:[7,20,26,30],direct:18,directli:35,directori:[19,21,22],discharg:[7,25,26],discharge_energi:7,discharging_r:7,discharging_rate_modifi:7,discount:[19,35],discret:[7,25],discuss:[18,27,35],displai:[19,34],diverg:35,do_nothing_act:17,doc:9,document:[10,18],doe:35,doesn:30,doi:[17,31],done:[7,30],dqn:11,dropdown:34,druri:31,dure:[17,34,35],dynam:17,e:[20,30,34],each:[5,6,7,17,19,27,30,34,35],ec:30,edit:35,eff_c:7,eff_d:7,effect:[17,30,32],effici:[17,30,31,34],electr:[17,29,30],electron:17,elev:22,elk:17,els:30,emiss:34,enabl:[19,23,27,35],end:[7,17,30,34],energi:[7,17,20,25,26,27,29,30,31,34,35],energy_efficiency_reward:[17,30],energy_lb:17,energy_pue_reward:[17,30],energy_ub:17,energy_usag:[17,30],energyplu:[17,19,23,35],engin:17,enivron:19,ensur:18,entropi:35,entropy_coeff:35,env:[5,6,9,11,12,17,30],env_config:[5,6,7,30],env_context:[5,6],env_index:[17,34],envconfig:5,envcontext:[5,6],environ:[5,6,7,17,21,22,23,28,29,30,31,32,34,35],environm:7,environment:34,episod:[7,17,19,34],episode_length_in_tim:[7,17],episode_reward_mean:[19,34],eplu:[19,30,35],epw:[5,9,17,19,35],equip:[7,17,30],essenti:32,estim:[27,30,35],et:17,eta:[19,30],etc:30,evalu:[27,34],evaluate_model:[12,32],evapor:27,exampl:[30,34,35],execut:[19,21,22,24,35],experi:19,explain:18,explicitli:11,extend:17,extern:[7,17,26,27,29,30],f2c:17,facil:29,factor:[19,30,35],fahrenheit:17,fail:30,fairchild:9,fals:[5,7,17,30,35],fan:[7,17,27,30],fashion:0,featur:18,feedback:[18,30],few:19,figur:26,file:[1,2,3,4,17,19,27,32,35],filenam:[17,19,35],find_and_init_capac:7,first:[17,19,20],fit:20,fix:[11,18],flag:[5,6,17,30],flexibl:[7,17,19,20,25,26,27,29,30,34],flexible_load:[5,19],flexible_workload_ratio:[7,17],flow:[7,27],focus:20,folder:[1,10,19,21],follow:[18,19,22,23,25,27,29,30,32,34,35],footprint:[7,20,26,30,34,35],forecast:17,fork:18,form:[17,29],format:[17,18,19,30,35],formula:17,found:[19,22,30,34,35],fraction:30,framework:[20,23,27,31],frederick:31,frequenc:17,frit:17,from:[7,17,18,19,21,22,25,27,30,34,35],full:27,full_load_pwr:7,futur:7,future_step:[7,17],gae:35,gamma:[19,35],gener:[17,35],geograph:17,get:[17,24,30],get_average_rack_fan_v:7,get_current_rack_load:7,get_default_config:11,get_default_policy_class:11,get_energy_vari:17,get_init_dai:17,get_ob:7,get_reward_method:17,get_total_ci:17,get_total_weath:17,get_total_wkl:17,getenv:30,git:[21,22],github:[10,21,22,27],gitignor:9,give:[34,35],given:[11,26,27,35],global:30,goal:[18,34],gradient:27,graph:34,greater:34,green:[18,19,22,23,26,27,30,33,34,35],grid:[26,27,30],gupta:31,gym:[17,20,23],gymnasium:[7,22],h2ocool:17,h:31,ha:[7,17],harb:31,have:[18,30,34],help:[18,34],helper_method:[9,12],henc:30,here:[18,22,24,30],hewlettpackard:[21,22],high:18,highest:27,hook:11,horizon:30,host:23,hour:[17,29,30],hourli:35,how:[30,33],how_is_the_function_cal:30,how_you_want_to_call_it:30,http:[17,21,22,24,31],hvac:[17,20,26,29,30],hvac_configur:27,hvac_pow:29,hvav:7,i:[17,20,22,30,34],ia2c:35,id:[0,7],identifi:[17,34],idl:[25,27,35],idle_pwr:7,ieee:17,igor:31,imag:22,implement:[30,34,35],improv:32,includ:[1,2,3,4,7,17,18,23,26,30,34,35],incorpor:20,increas:25,independ:[31,35],index:17,indic:[17,29,30,34],individu:[5,6,18,23],individual_reward_weight:[5,19,30],inf:30,influenc:17,info:[5,6,7,35],inform:[7,19,27,29,30,35],init_dai:17,initi:[7,17,34,35],inlet:[7,17,27],inlet_temp:7,inlet_temp_rang:27,inlud:30,input:[19,30],input_load:7,instal:[17,21,27],instanc:0,instanti:17,instruct:[22,23],integr:23,intend:18,intens:[1,2,7,17,19,26,27,29,30],interest:18,interfac:[20,22,23,30],intern:[7,26,29,31],intersocieti:17,intes:17,invers:30,involv:35,ippo:35,is_ful:7,isol:22,issu:11,istermin:17,it_equipment_energi:[17,30],it_equipment_pow:30,it_fan_airflow_ratio_lb:27,it_fan_airflow_ratio_ub:27,it_fan_full_load_v:27,it_pow:29,ite_load:[17,30],ite_load_pct:7,ite_load_pct_list:7,iter:[19,35],itfan:7,itfan_curve2:7,itfan_ref_p:27,itfan_ref_v_ratio:27,its:[17,32],j:[17,27],javier:31,jean:31,journal:[17,31],json:[9,27],juan:31,k:[27,31],kaiyu:17,kei:34,kennedi:[5,9,17,19,35],kg:27,kl:35,kl_coeff:35,knowledg:20,kw:30,kwarg:[11,17,34],kwh:30,last:[17,29],later:17,latest:[21,22],latest_experi:13,launch:[19,22,35],lawri:31,learn:[19,20,31,34],left:[17,29,34],legacy_callbacks_dict:17,length:[17,27],level:20,licens:9,light:11,limit:17,linda:31,linux:21,list:[7,17,19,30,34,35],load:[5,6,7,17,19,20,25,26,27,29,30,32,34,35],load_left_mean:34,local:[19,35],localhost:34,locat:[5,17,19,31,35],log:19,logdir:[19,34],logger:11,logger_cr:11,look:17,low:31,lower:[17,30],lower_lim_u:7,lower_lim_v:7,lower_u:7,lower_v:7,lowest:27,lr:35,ls:[19,30],ls_action:30,ls_norm_load_left:30,ls_original_workload:30,ls_penalty_flag:30,ls_reward:[5,6,30],ls_shifted_workload:30,ls_unasigned_day_load_left:30,luca:31,m:[17,27],machineri:31,maddpg:[9,12,35],maddpgconfig:11,maddpgconfigst:11,maddpgstabl:11,made:18,mai:27,main:[20,27],maintain:18,make:[7,26,29,32],make_bat_fwd_env:17,make_dc_env:17,make_dc_pyeplus_env:17,make_env:[9,12],make_envs_pyenv:[9,12],make_ls_env:17,makoviichuk:31,makoviychuk:31,manag:[12,29],maneg:17,manjavaca:31,manner:17,manufactur:27,marl:[19,20,27,35],match:30,max:17,max_bat_cap_mw:[5,17,19],max_bsiz:7,max_temp:[7,30],max_v:17,max_w_per_rack:7,maxim:30,maximum:[17,27,30],md:9,mechan:[20,27,30],megawatt:17,member:10,memori:19,merg:18,messag:18,method:[11,17,30,34,35],metric:[17,20,33],miguel:31,min_temp:[7,30],min_v:17,mingfei:31,minim:30,minimum:[17,30],miss:24,mix:31,model:[7,17,19,20,23,26,27,33,34,35],modifi:[27,30,34,35],modifii:32,modul:[12,20,32,34],modular:20,molina:31,monitor:[32,33],month:17,monthan:9,mordatch:31,more:[23,35],multi:[11,20,27,31,35],multi_agent_batch:11,multi_agent_env:[5,6],multiagentdict:[5,6],multiagentenv:[5,6],multiagentepisod:17,multipl:[19,20],must:[19,27,30,35],mwh:30,n:[21,29,30],n_step:17,n_vars_batteri:[7,17],n_vars_energi:[7,17],name:[13,17,19,30,35],nan:19,navig:[22,34],necessari:[18,32],need:[20,24,31],neg:30,niev:31,nois:17,none:[5,6,7,11,17,30,34],norm_carbon:29,norm_ci:[17,30],norm_load_left:[17,29],normal:[7,17,29,30],normalis:29,normalized_observ:7,normalizeobserv:7,noth:[17,25],num_ag:19,num_rack:7,num_racks_per_row:27,num_row:27,num_sin_cos_var:17,num_work:19,number:[17,19,27],ny:[5,17,19,31,35],nyi:17,nyis_ng_:[5,9,19,35],o:[30,31],ob:[5,6,7],object:[7,17,20,27,30],observ:[5,6,7,17,28],observation_spac:7,observation_vari:7,obsev:7,obtain:[7,17,30,35],obtain_path:17,off:17,on_episode_end:[17,34],on_episode_start:[17,34],on_episode_step:[17,34],onc:[18,34],one:[19,34,35],onli:17,open:[18,35],openai:[20,23],oper:[26,35],optim:[17,19,20,26,27,30,35],optimal_temperature_rang:[17,30],option:[5,6,7,11,17,23,30],orderli:0,org:31,organ:1,origin:[11,30],os:[21,30],other:[20,23,27,30],our:20,out:[0,19],out_of_tim:17,outlet:7,outlin:32,output:19,output_load:7,outsid:[7,30],outside_temp:30,over:[11,34,35],overal:[18,34],overrid:[13,17],own:20,pacif:31,packag:12,panda:[7,17],parallel:19,param:[5,6,7,17,30],paramet:[0,5,6,7,17,30,35],part:17,particular:17,pass:17,pattern:27,pd:17,pedersen:31,penalti:[17,30],perform:[17,27,32,34],period:[29,35],phenomena:17,philip:31,piec:30,pieter:31,pip:[21,24],placehold:17,plan:34,plu:31,plug:23,point:[17,32],polici:[11,17,34,35],policyspec:11,possibl:30,postema:17,power:[7,17,26,27,30,34],ppo:[19,35],ppo_exampl:9,ppoconfig:30,practic:18,pre:20,prefix:22,previou:34,price:30,print:0,prior:11,priorit:20,privileg:22,probabl:24,problem:[20,35],procedur:32,proceed:31,process:[17,32,34,35],produc:[7,34],profil:27,program:31,progress:[0,34],project:18,proper:17,properli:18,prototyp:17,provid:[11,18,19,20,22,23,27,30,32,34,35],pue:[17,30],pull:[18,22],py:[9,19,30,32,34,35],pyeplus_callback:17,python:[19,21,22,35],qualiti:18,r:21,r_:30,raboso:31,rack:[7,23,27],rack_config:7,rack_cpu_config:7,rack_return_approach_temp_list:[7,27],rack_supply_approach_temp_list:[7,27],rackwis:7,rackwise_cpu_pwr:7,rackwise_itfan_pwr:7,rackwise_outlet_temp:7,raghunathan:17,rai:[0,5,6,11,13,17],rais:[0,17],random:[5,6,7],rang:[7,25,27],rate:[7,17,19,27,29,35],ratio:[7,17,19,27,30],raw_config:5,raw_curr_st:7,reach:17,read:[17,30],readm:9,real:27,recommend:22,reduc:[25,35],reduct:20,refer:[17,27,30],regist:0,regular:35,reinforc:[20,31,34],releas:25,relev:[18,29,34],renew:[17,30],renewable_energy_ratio:[17,30],renewable_energy_reward:[17,30],replai:19,repositori:[10,18,27,35],repres:[19,34,35],request:18,requir:[9,21,27,30,35],reset:[5,6,7,17],respect:35,result:[9,13,32,33],results_dir:13,review:18,rew:[5,6],reward:[5,6,7,17,19,20,28],reward_cr:[9,12,30],reward_info:30,reward_method:17,reward_method_map:30,reward_param:30,rho_air:27,rise:17,rl:[9,17,21,22,30,32,35],rllib:[0,5,6,11,13,17],rllib_callback:[9,12,34],robust:20,roll:19,romero:31,room:[7,27,29,30],row:27,rtype:7,run:[8,19,22,32,34,35],run_id:8,ryan:31,s:[7,17,18,23,27,30,31,32,34],same:[27,30],sampl:19,save:35,sc_ob:17,scalabl:20,scale:[17,30],schedul:[26,30],schema:30,scheme:30,schroeder:31,scienc:17,script:[19,22,30,35],section:[19,22,23,27,35],see:24,seed:[5,6,7],seek:30,select:[34,35],self:[5,6,23,34],separ:17,server:[17,23],server_characterist:27,set:[7,17,27],set_ambient_temp:7,set_bat_soc:7,set_dcload:7,set_shifted_wkld:7,setpoint:[7,26,29,30],shape:[20,30],share:[29,30],sheet:27,shift:[5,6,17,19,20,25,26,29,30,34,35],shimon:31,shm:22,should:[17,18,19,24,30,35],show:30,shown:26,sign:30,signal:[7,17,29],sim_battery_247:7,simplest:19,simul:[19,23,27,31,35],sin:17,sinc:30,sine:[17,29],sine_day_of_year:29,sine_time_hour:29,sinergym:[19,22,31,35],singl:[11,35],size:[19,22,25,35],soc:34,solana:31,some:[17,18,22,24,30],sourc:[0,5,6,7,8,11,13,17,18,25,29,30,35],space:[7,25],spatial:27,spec:27,specif:[17,24,27,34,35],specifi:[19,20,23,30,35],spent:[17,30],sphinx:9,ssh:21,stabil:11,standalon:17,standalone_pyeplu:17,standar:17,standard:[17,18,20],starcraft:31,start:[17,18,30,34,35],start_month:17,state:[5,6,7,17,29,30],statespac:17,step:[5,6,7,17,25,29,30,32,34],step_count:34,storag:26,store:[17,19,25,34],str:[0,7,13,17],string:[0,7,17,35],structur:10,studi:31,style:18,sub:19,submit:18,submodul:12,subpackag:12,subsect:27,subset:17,successfulli:22,sudo:22,suit:27,sun:[17,31],suppli:20,support:[17,20,22,35],system:[17,26,31,34],t:[17,22,30],t_u:7,tabl:[25,29,35],take:[7,17,30],taken:17,tamar:31,tarun:31,task:[17,35],temp_column:17,temp_stat:7,temperatur:[7,17,25,26,27,29,30],temperature_efficiency_reward:[17,30],templat:17,tensorboard:[19,32,33],term:17,termin:[5,6,17,34],test:[9,17,18,19,20],test_mod:7,test_util:9,text:17,them:20,therefor:30,thermal:[17,27],thermomechan:17,thi:[1,11,17,18,19,24,27,30,34,35],thoma:17,three:[1,20,25,26,27,29,30,35],through:[20,23],time:[7,17,25,27,29,30],time_manag:17,timedelta:[7,17],timelin:17,tip:23,tool:[24,34],torr:31,total:[7,17,27,29,30,34],total_battery_soc:34,total_datacenter_full_load:7,total_energy_consumpt:[17,30],total_energy_with_batteri:[17,34],total_pow:29,total_power_consumpt:30,tou:[17,30],tou_reward:[17,30],tower:[7,17,27],track:34,trade:17,train:[0,9,12,17,30,31,32,35],train_a2c:[9,12,35],train_batch_s:[11,35],train_maddpg:[9,12,35],train_one_step:11,train_ppo:[9,12,19,30,35],trainabl:0,training_step:11,transit:17,transport:31,trial:19,trigger:17,truncat:[5,6],tslib:[7,17],tune:11,tupl:[17,30],turkish:17,turn:17,two:[30,35],txt:[9,21],type:[0,5,6,7,11,17,20,25,29],typeerror:0,typic:26,ubuntu:21,ul:30,unassign:[30,34],under:[17,18,19,26,27,35],uninterrupt:26,union:[0,5,6,13],unit:[7,25],until:19,up:26,updat:[7,17,30],update_ci:7,update_st:7,update_workload:7,upload:35,upper:[17,30],upper_lim_u:7,upper_lim_v:7,upper_u:7,upper_v:7,url:[31,34],us:[11,17,19,20,21,22,23,25,30,32,33,35],usa:31,usa_az_tucson:9,usa_ny_new:[5,9,17,19,35],usa_wa_port:9,usag:[7,17,19,27,29,30],use_ga:35,use_local_crit:35,use_ls_cpu_load:17,useful:23,user:[11,20,23,27,30,35],user_data:[17,34],util:[9,12,20,27,30,34],utils_cf:[9,12],v3:22,v:[17,22],valid:11,valu:[7,17,19,27,29,30,34,35],valueerror:17,variabl:[17,27,29,30,32,35],varieti:18,variou:[17,34],veloc:[7,27],version:[21,22,24],versu:20,view:[19,27,34],viktor:31,virtual:17,visibl:30,visual:[19,32,34],visualstudio:24,vk:17,voltag:17,w:[27,30],wa:35,waat_ng_:9,wai:[34,35],want:30,we:[17,30,34],weather:[1,9,12,17,19,26,27,29],weather_fil:[5,35],weather_filenam:17,weather_manag:17,websit:24,weight:[17,19,30],welcom:18,well:20,were:18,what:[18,24,35],when:[18,26],where:[30,35],whether:17,which:[11,17,27,34],whiteson:31,whole:[7,29],why:18,winkelmann:31,within:[17,23,30],witt:31,work:[17,18,21,30],worker:[17,34],workload:[1,7,9,12,17,19,26,27,30,31,34],workload_fil:[5,35],workload_filenam:17,workload_manag:17,workshop:31,wrap:[0,22],wrapper:[11,17],write:18,wu:31,year:[17,30,35],yi:31,york:[5,9,17,19,31,35],you:[18,24,31,32,34],your:[18,24,34],yue:31,zheng:31,zone_air_temp:29,zone_air_therm_cooling_stpt:29},titles:["create_trainable module","data package","data.CarbonIntensity package","data.Weather package","data.Workload package","dcrl_env module","dcrl_eplus_env module","envs package","evaluate_model module","Folder Structure","Code","maddpg module","Class and member documentation","train module","train_a2c module","train_maddpg module","train_ppo module","utils package","Contribution Guidelines","Getting Started","DCRL-Green","DCRL-Green\u2019s custom DC simulation (DCRL)","EnergyPlus enabled DC simulation (DCRLeplus)","Installation","Usefull tips for the installation","Actions","Agents","Environment","Overview","Observations","Reward function","References","How to evaluate DCRL-Green model","Train and Evaluate Statistics","How to Monitor Training Results Using TensorBoard","Usage"],titleterms:{"0":24,"1":[19,22,24,27],"14":24,"2":[19,22,24,27],"3":[19,27],"class":12,"function":30,action:25,add:34,agent:[19,26,35],algorithm:35,base_ag:17,bat_env_fwd_view:7,battery_model:7,build:24,c:24,carbon:35,carbon_l:7,carbonintens:2,cchardet:24,characterist:27,code:10,collabor:30,configur:[27,35],contain:22,content:[1,2,3,4,7,17],contribut:18,create_train:0,custom:[21,30,34],data:[1,2,3,4,35],datacent:7,dc:[21,22,27],dc_config:17,dc_config_read:17,dc_gym:7,dcrl:[19,20,21,32],dcrl_env:5,dcrl_eplus_env:6,dcrleplu:[19,22],depend:21,docker:22,document:12,enabl:22,energyplu:22,env:7,environ:[19,27],error:24,evalu:[32,33],evaluate_model:8,exampl:19,fail:24,first:[21,22],folder:9,geometri:27,get:19,greater:24,green:[20,21,32],guidelin:18,helper_method:17,how:[32,34],hvac:27,hyperparamet:35,independ:30,instal:[22,23,24],intens:35,learn:35,maddpg:11,make_env:17,make_envs_pyenv:17,manag:17,manual:22,member:12,metric:34,microsoft:24,model:32,modul:[0,1,2,3,4,5,6,7,8,11,13,14,15,16,17],monitor:[19,34],observ:29,overview:28,packag:[1,2,3,4,7,17],prerequisit:22,refer:31,reinforc:35,requir:24,result:[19,34],reward:30,reward_cr:17,rllib_callback:17,s:21,server:27,setup:[21,22],simul:[21,22],start:19,statist:33,structur:9,submodul:[7,17],subpackag:1,tensorboard:34,time:[21,22],tip:24,train:[13,19,33,34],train_a2c:14,train_maddpg:15,train_ppo:16,us:34,usag:35,useful:24,util:17,utils_cf:17,visual:24,weather:[3,35],wheel:24,workload:[4,35]}})