#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace std;
 
int main()
{
	string fname = "/home/research-student/omnet-fanet/data-processing-scripts/Stationary_Test_Dataset_NP10000_BPSK_6-5Mbps_downlink_RESULTS_nn_V2-2.csv";
 
	vector<vector<string>> content;
	vector<string> row;
	string line, word;
 
	fstream file (fname, ios::in);
	if(file.is_open())
	{   
	    getline(file, line);
		while(getline(file, line))
		{
			row.clear();
 
			stringstream str(line);
 
			while(getline(str, word, ','))
				row.push_back(word);
			content.push_back(row);
		}
	}
	else
		cout<<"Could not open the file\n";
	cout << setprecision(9) << fixed;

	for(int i=0;i<content.size();i++)
	{
		cout << std::stoi(content[i][0]) + std::stod(content[i][1]) << "\n";
		// for(int j=0;j<content[i].size();j++)
		// {
		// 	cout<<content[i][j]<<" ";
		// }
		// cout<<"\n";
	}
 
	return 0;
}
 