

<!-- TABLE_START -->

| log                     | backbone   | categorical_features     | continuous_features            |   val_f1_mean |
|:------------------------|:-----------|:-------------------------|:-------------------------------|--------------:|
| BPI12                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time', 'amount'] |        0.6963 |
| BPI12                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.7073 |
| BPI12                   | gpt2-small | ['activity', 'resource'] | ['amount']                     |        0.6915 |
| BPI12                   | gpt2-small | ['activity']             | ['accumulated_time', 'amount'] |        0.665  |
| BPI12                   | gpt2-small | ['activity']             | ['accumulated_time']           |        0.6654 |
| BPI12                   | gpt2-small | ['activity']             | ['amount']                     |        0.5346 |
| BPI12                   | gpt2-small | ['activity']             | []                             |        0.6491 |
| BPI15                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.1121 |
| BPI15                   | gpt2-small | ['activity', 'resource'] | []                             |        0.1205 |
| BPI15                   | gpt2-small | ['activity']             | ['accumulated_time']           |        0.1103 |
| BPI15                   | gpt2-small | ['activity']             | []                             |        0.1286 |
| BPI17                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time', 'amount'] |        0.7114 |
| BPI17                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.7082 |
| BPI17                   | gpt2-small | ['activity', 'resource'] | ['amount']                     |        0.7079 |
| BPI17                   | gpt2-small | ['activity']             | ['accumulated_time', 'amount'] |        0.6742 |
| BPI17                   | gpt2-small | ['activity']             | ['accumulated_time']           |        0.676  |
| BPI17                   | gpt2-small | ['activity']             | ['amount']                     |        0.6825 |
| BPI17                   | gpt2-small | ['activity']             | []                             |        0.683  |
| BPI19                   | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.2452 |
| BPI19                   | gpt2-small | ['activity', 'resource'] | []                             |        0.2404 |
| BPI19                   | gpt2-small | ['activity']             | ['accumulated_time']           |        0.178  |
| BPI19                   | gpt2-small | ['activity']             | []                             |        0.198  |
| BPI20PrepaidTravelCosts | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.5069 |
| BPI20PrepaidTravelCosts | gpt2-small | ['activity', 'resource'] | []                             |        0.5231 |
| BPI20PrepaidTravelCosts | gpt2-small | ['activity']             | ['accumulated_time']           |        0.503  |
| BPI20PrepaidTravelCosts | gpt2-small | ['activity']             | []                             |        0.5205 |
| BPI20RequestForPayment  | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.5588 |
| BPI20RequestForPayment  | gpt2-small | ['activity', 'resource'] | []                             |        0.563  |
| BPI20RequestForPayment  | gpt2-small | ['activity']             | ['accumulated_time']           |        0.5753 |
| BPI20RequestForPayment  | gpt2-small | ['activity']             | []                             |        0.5834 |
| BPI20TravelPermitData   | gpt2-small | ['activity', 'resource'] | ['accumulated_time']           |        0.3635 |
| BPI20TravelPermitData   | gpt2-small | ['activity', 'resource'] | []                             |        0.4282 |
| BPI20TravelPermitData   | gpt2-small | ['activity']             | ['accumulated_time']           |        0.351  |
| BPI20TravelPermitData   | gpt2-small | ['activity']             | []                             |        0.4273 |

<!-- TABLE_END -->
