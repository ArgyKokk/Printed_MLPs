#!/bin/bash

hweval () {
	make dcsyn
	make sta
	./extra_scripts/eval.sh 
	make power

}

#make sure that ./extra_scripts/eval.sh registers the accuracy. The following commands might not be needed.
sed -i '/^python3/d' ./extra_scripts/eval.sh
echo "python3 \$dir/accuracy.py \$dir/../sim/sim.Ytest \$dir/../sim/output.txt | tee \$dir/../reports/accuracy.rpt" >> ./extra_scripts/eval.sh
volt1dir="1V"
volt06dir="0.6V"

verilogdir="only_quantization"
mkdir ./only_quantization_reports
mkdir ./only_quantization_reports/$volt1dir
mkdir ./only_quantization_reports/$volt06dir

for verilog in $(ls $verilogdir)
do
        #copy the test dataset in the sim directory
        n="${verilog#*_}"
        n="${n%_*}"
        id="$(echo $n | tr -dc '0-9')"
        echo $id
        echo "Pareto"
        filename="sim.Xtest"$id
        cp ./datasetsQ/$filename ./sim
        mv ./sim/$filename ./sim/"sim.Xtest"

        #extract the input decimals
        name="${verilog#*_}"
        name="${name#*_}"
        dec=${name:5:1}
        
        #change the input value
        sed -i -e "s/SIM_WIDTH_A=.*/SIM_WIDTH_A=${dec}/" ./extra_scripts/circuit_conf.sh 
        
	#copy verilog file
	echo $verilog
        echo $filename
	name="${verilog%.*}"
	cp $verilogdir/$verilog hdl/top.v
        
	#set 1V lib
	sed -i "/export ENV_LIBRARY_DB=/c\export ENV_LIBRARY_DB=\"EGFET_1.0V.db\"" scripts/env.sh
	#grep "ENV_LIBRARY_DB" scripts/env.sh

	#evaluate
	hweval
	mv reports reports.1p0V.${name}
        mv reports.1p0V.${name} ./only_quantization_reports/$volt1dir


	#you could do the same for 0.8V also: EGFET_0.8V.db
done
