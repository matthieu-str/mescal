## MAIN SETS: Sets whose elements are input directly in the data file
set YEARS;
set PERIODS; # time periods
set SECTORS; # sectors of the energy system
set END_USES_INPUT; # Types of demand (end-uses). Input to the model
set END_USES_CATEGORIES; # Categories of demand (end-uses): electricity, heat, mobility
set END_USES_TYPES_OF_CATEGORY {END_USES_CATEGORIES}; # Types of demand (end-uses).
set RESOURCES; # Resources: fuels (wood and fossils) and electricity imports 
set END_USES_TYPES := setof {i in END_USES_CATEGORIES, j in END_USES_TYPES_OF_CATEGORY [i]} j; # secondary set
set TECHNOLOGIES_OF_END_USES_TYPE {END_USES_TYPES}; # set all energy conversion technologies (excluding storage technologies)
set INFRASTRUCTURE; # Infrastructure: DHN, grid, and intermediate energy conversion technologies (i.e. not directly supplying end-use demand)
set PV_TECH;

## SECONDARY SETS: a secondary set is defined by operations on MAIN SETS
set LAYERS := RESOURCES union END_USES_TYPES; # Layers are used to balance resources/products in the system
set TECHNOLOGIES := (setof {i in END_USES_TYPES, j in TECHNOLOGIES_OF_END_USES_TYPE [i]} j) union INFRASTRUCTURE;
set TECHNOLOGIES_OF_END_USES_CATEGORY {i in END_USES_CATEGORIES} within TECHNOLOGIES := setof {j in END_USES_TYPES_OF_CATEGORY[i], k in TECHNOLOGIES_OF_END_USES_TYPE [j]} k;

### PARAMETERS ###
param end_uses_demand_year {END_USES_INPUT, SECTORS, YEARS} >= 0 default 0; # end_uses_year: table end-uses demand vs sectors (input to the model). Yearly values.
param end_uses_input {i in END_USES_INPUT, y in YEARS} := sum {s in SECTORS} (end_uses_demand_year [i,s,y]); # Figure 1.4: total demand for each type of end-uses across sectors (yearly energy) as input from the demand-side model
param i_rate > 0; # discount rate (real discount rate)
param t_op {PERIODS}; # duration of each time period [h]
param total_time := sum {t in PERIODS} (t_op [t]); # added just to simplify equations
param lighting_month {PERIODS} >= 0, <= 1; # %_lighting: factor for sharing lighting across months (adding up to 1)
param cooling_month {PERIODS} >= 0, <= 1; # %_sc: factor for sharing space cooling across months (adding up to 1)

# f: input/output Resources/Technologies to Layers. Reference is one unit ([GW] or [Mpkm/h] or [Mtkm/h]) of (main) output of the resource/technology. input to layer (output of technology) > 0.
param layers_in_out {RESOURCES union TECHNOLOGIES, LAYERS, YEARS} default 0;

# Attributes of TECHNOLOGIES
param ref_size {TECHNOLOGIES} >= 0 default 0.001; # f_ref: reference size of each technology, expressed in the same units as the layers_in_out table. Refers to main output (heat for cogen technologies). storage level [GWh] for STORAGE_TECH
param c_inv {TECHNOLOGIES,YEARS} >= 0 default 0.000001; # Specific investment cost [MCHF/GW].[MCHF/GWh] for STORAGE_TECH
param c_maint {TECHNOLOGIES,YEARS} >= 0 default 0; # O&M cost [MCHF/GW/year]: O&M cost does not include resource (fuel) cost. [MCHF/GWh] for STORAGE_TECH
param lifetime {TECHNOLOGIES,YEARS} >= 0 default 20; # n: lifetime [years]
param f_max {TECHNOLOGIES} >= 0 default 300000; # Maximum feasible installed capacity [GW], refers to main output. storage level [GWh] for STORAGE_TECH
param f_min {TECHNOLOGIES} >= 0 default 0; # Minimum feasible installed capacity [GW], refers to main output. storage level [GWh] for STORAGE_TECH
param fmax_perc {TECHNOLOGIES} >= 0, <= 1 default 1; # value in [0,1]: this is to fix that a technology can at max produce a certain % of the total output of its sector over the entire year
param fmin_perc {TECHNOLOGIES} >= 0, <= 1 default 0; # value in [0,1]: this is to fix that a technology can at min produce a certain % of the total output of its sector over the entire year
param fmax_perc_mob {TECHNOLOGIES} >= 0, <= 1 default 1; # value in [0,1]: this is to fix that a technology can at max produce a certain % of the total output of its sector over the entire year
param fmin_perc_mob {TECHNOLOGIES} >= 0, <= 1 default 0; # value in [0,1]: this is to fix that a technology can at min produce a certain % of the total output of its sector over the entire year
param c_p_t {TECHNOLOGIES, PERIODS} >= 0, <= 1 default 1; # capacity factor of each technology and resource, defined on monthly basis. Different than 1 if F_Mult_t (t) <= c_p_t (t) * F_Mult
param c_p {TECHNOLOGIES} >= 0, <= 1 default 1; # capacity factor of each technology, defined on annual basis. Different than 1 if sum {t in PERIODS} F_Mult_t (t) * t_op (t) <= c_p * F_Mult
param tau {i in TECHNOLOGIES, y in YEARS} := i_rate * (1 + i_rate)^lifetime [i,y] / (((1 + i_rate)^lifetime [i,y]) - 1); # Annualisation factor for each different technology
param trl {TECHNOLOGIES} >=0 default 9; # Technology Readiness Level

# Attributes of RESOURCES
param c_op {RESOURCES,PERIODS,YEARS} >= 0 default 0.000001; # cost of resources in the different periods [MCHF/GWh]
param avail {RESOURCES,YEARS} >= 0 default 0; # Yearly availability of resources [GWh/y]

# Other parameters 
param trl_min default 1;
param trl_max default 9;


## VARIABLES [Tables 1.2, 1.3] ###
var End_Uses {LAYERS, PERIODS, YEARS} >= 0; # total demand for each type of end-uses (monthly power). Defined for all layers (0 if not demand)

var F_Mult {TECHNOLOGIES, YEARS} >= 0; # F: installed size, multiplication factor with respect to the values in layers_in_out table
var F_Mult_t {RESOURCES union TECHNOLOGIES, PERIODS, YEARS} >= 0; # F_t: Operation in each period. multiplication factor with respect to the values in layers_in_out table. Takes into account c_p
var C_inv {TECHNOLOGIES, YEARS} >= 0; # Total investment cost of each technology
var C_maint {TECHNOLOGIES, YEARS} >= 0; # Total O&M cost of each technology (excluding resource cost)
var C_op {RESOURCES, YEARS} >= 0; # Total O&M cost of each resource
var TotalCost {YEARS} >= 0; # C_tot: Total GWP emissions in the system [ktCO2-eq./y]

# variables added for recording the output results 
var Monthly_Prod{TECHNOLOGIES,PERIODS,YEARS}; #[GWh] the production in each time period except storage technologies
var Annual_Prod{TECHNOLOGIES,YEARS};
var Annual_Res{RESOURCES,YEARS};

### CONSTRAINTS ###

## End-uses demand calculation constraints 

# From annual energy demand to monthly power demand. End_Uses is non-zero only for demand layers.
subject to end_uses_t {l in LAYERS, t in PERIODS, y in YEARS}:
	End_Uses [l,t,y] = (if l == "ELECTRICITY" then
			(end_uses_input[l,y] / total_time + end_uses_input["LIGHTING",y] * lighting_month [t] / t_op [t] + end_uses_input["HEAT_LOW_T_SC",y] * cooling_month [t] / t_op [t])
		else 
			0 ); # For all layers which don't have an end-use demand


## Layers

# Layer balance equation with storage. Layers: input > 0, output < 0. Demand > 0. Storage: in > 0, out > 0;
# output from technologies/resources/storage - input to technologies/storage = demand. Demand has default value of 0 for layers which are not end_uses
subject to layer_balance {l in LAYERS, t in PERIODS, y in YEARS}:
	0 = (sum {i in RESOURCES union TECHNOLOGIES} (layers_in_out[i, l, y] * F_Mult_t [i, t, y])
		- End_Uses [l, t, y]
		);


# For avoiding F_Mult[i] tends to be a large number while all F_Mult_t[i,t] = 0
subject to f_mult_prevention{i in TECHNOLOGIES, y in YEARS}:
	F_Mult[i,y]<=1000000 * sum{t in PERIODS} F_Mult_t[i,t,y];


# min & max limit to the size of each technology
subject to size_limit {i in TECHNOLOGIES, y in YEARS}:
	f_min [i] <= F_Mult [i,y] <= f_max [i];
	
# relation between mult_t and mult via period capacity factor. This forces max monthly output (e.g. renewables)
subject to capacity_factor_t {i in TECHNOLOGIES, t in PERIODS, y in YEARS}:
	F_Mult_t [i, t, y] <= F_Mult [i, y] * c_p_t [i, t];


# relation between mult_t and mult via yearly capacity factor. This one forces total annual output
subject to capacity_factor {i in TECHNOLOGIES, y in YEARS}: #for ccus technology, unit: F_Mult_t: t/h
	sum {t in PERIODS} (F_Mult_t [i, t, y] * t_op [t]) <= F_Mult [i,y] * c_p [i] * total_time;	


## Resources

# [Eq. 1.12] Resources availability equation
subject to resource_availability {i in RESOURCES, y in YEARS}:
	sum {t in PERIODS} (F_Mult_t [i, t, y] * t_op [t]) <= avail [i,y];

## Additional constraints: Constraints needed for the application to Switzerland (not needed in standard MILP formulation)

# [Eq 1.22] Definition of min/max output of each technology as % of total output in a given layer. 
# Normally for a tech should use either f_max/f_min or f_max_%/f_min_%
subject to f_max_perc {i in END_USES_TYPES, j in TECHNOLOGIES_OF_END_USES_TYPE[i], y in YEARS}:
	sum {t in PERIODS} (F_Mult_t [j, t, y] * t_op[t]) <= fmax_perc [j] * sum {j2 in TECHNOLOGIES_OF_END_USES_TYPE[i], t2 in PERIODS} (F_Mult_t [j2, t2, y] * t_op [t2]);

subject to f_min_perc {i in END_USES_TYPES, j in TECHNOLOGIES_OF_END_USES_TYPE[i], y in YEARS}:
	sum {t in PERIODS} (F_Mult_t [j, t, y] * t_op[t])  >= fmin_perc [j] * sum {j2 in TECHNOLOGIES_OF_END_USES_TYPE[i], t2 in PERIODS} (F_Mult_t [j2, t2, y] * t_op [t2]);

## Cost

# [Eq. 1.3] Investment cost of each technology
subject to investment_cost_calc_1 {i in TECHNOLOGIES, y in YEARS}:
	C_inv [i,y] = c_inv [i,y] * F_Mult [i,y];

# [Eq. 1.4] O&M cost of each technology
subject to main_cost_calc {i in TECHNOLOGIES, y in YEARS}:
	C_maint [i,y] = c_maint [i,y] * F_Mult [i,y];		

# [Eq. 1.10] Total cost of each resource
subject to op_cost_calc {i in RESOURCES, y in YEARS}:
	C_op [i,y] = sum {t in PERIODS} (c_op [i,t,y] * F_Mult_t [i, t, y] * t_op [t]);

# [Eq. 1.1]	
subject to totalcost_cal{y in YEARS}:
	TotalCost[y] = sum {i in TECHNOLOGIES} (tau [i,y] * C_inv [i,y] + C_maint [i,y]) + sum {j in RESOURCES} C_op [j,y] ;

# Only for saving data
subject to production{i in TECHNOLOGIES, t in PERIODS, y in YEARS}:
	Monthly_Prod[i,t,y]=F_Mult_t[i,t,y]*t_op[t];

subject to production2{i in TECHNOLOGIES, y in YEARS}:
	Annual_Prod[i,y]=sum{t in PERIODS} Monthly_Prod[i,t,y];

subject to res_import{r in RESOURCES, y in YEARS}:
	Annual_Res[r,y] = sum{t in PERIODS} F_Mult_t[r,t,y]*t_op[t];

# TRL choice
subject to trl_choice{y in YEARS, i in TECHNOLOGIES: trl[i]>trl_max or trl[i]<trl_min}:
	F_Mult[i,y]=0;

subject to battery{y in YEARS}:
	F_Mult["BATTERY",y]=0.3*(F_Mult["PV",y] + F_Mult["WIND_ONSHORE",y]);

# subject to extra_grid:
#	F_Mult ["GRID"] = 1 + (9400 / c_inv["GRID"]) * (F_Mult ["WIND_ONSHORE"] + F_Mult ["PV"]) / (f_max ["WIND_ONSHORE"] + f_max ["PV"]);