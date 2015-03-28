<?php
echo "Type in your variable (L70P is a popular choice): ";
$handle = fopen ("php://stdin","r");
$variant = fgets($handle);
echo "What is the category split? (2, 22, 3, and 4 are options): ";
$handle2 = fopen ("php://stdin","r");
$category = fgets($handle2);

$python_sc = '/opt/predict/classifyMutation.py ';

    if (empty($category)) {
        $command = escapeshellcmd($python_sc . $variant);
    } else {
        $command = escapeshellcmd($python_sc . $variant . " -c " . $category);
    }
    
$output = shell_exec($command);
echo $output;

?>
