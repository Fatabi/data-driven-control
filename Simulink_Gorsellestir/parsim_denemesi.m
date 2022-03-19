model  = 'sarkac_RL';
load_system(model);
in     = Simulink.SimulationInput(model);
in     = in.setModelParameter('SimulationMode', 'rapid-accelerator');
in     = in.setModelParameter('RapidAcceleratorUpToDateCheck', 'off');
for i = 1:100
in(i) = Simulink.SimulationInput(model) ; 
end
simOut = parsim(in,'ShowProgress', 'on') ;