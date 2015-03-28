<?php 
echo "Type in your variable (L70P is a popular choice): ";
$handle = fopen ("php://stdin","r");
$input_var = fgets($handle);
echo "What is the category split? (2, 22, 3, and 4 are options): ";
$handle2 = fopen ("php://stdin","r");
$input_var2 = fgets($handle2);
$input_var .= " -c ";
$input_var .= $input_var2;
$python_sc = '/opt/predict/classifyMutation.py ';
$command = escapeshellcmd($python_sc . $input_var);
$output = shell_exec($command);
echo $output;

?>
