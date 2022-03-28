// To run this code after installing the PSDD C++ library run:
// g++ -no-pie paths_psdd_mult.cpp src/psdd_manager.cpp src/psdd_node.cpp src/psdd_parameter.cpp src/psdd_unique_table.cpp src/random_double_generator.cpp -Iinclude -Llib/linux -lsdd -lgmp

// change the input files approprietly

#include <iostream>
#include <unordered_map>
#include <stdio.h>
#include <dirent.h>
// #include <psdd/cnf.h>
// #include <psdd/optionparser.h>
#include <psdd/psdd_manager.h>
#include <psdd/psdd_unique_table.h>
#include <psdd/psdd_node.h>
#include <psdd/psdd_parameter.h>
#include <psdd/random_double_generator.h>
extern "C" {
#include <sdd/sddapi.h>
}
using namespace std;

int main()
{
    string vtree_name = "/home/kushagra/psdd-all/5x5_icaps.vtree";
	string sdd1_name = "/home/kushagra/psdd-all/5x5_icaps.sdd";
	// string psdd1_name = "/home/kushagra06/psdd-master/coverage.psdd";
	string sdd2_name = "/home/kushagra/psdd-all/5x5_icaps_l5.sdd";
	// string sdd3_name = "/home/kushagra06/psdd-master/coverage_landmarks_constr.sdd";


	Vtree* vtree = sdd_vtree_read(vtree_name.c_str());
	SddManager* sdd_manager = sdd_manager_new(vtree);
	SddNode* sdd1 = sdd_read(sdd1_name.c_str(), sdd_manager);
	SddNode* sdd2 = sdd_read(sdd2_name.c_str(), sdd_manager);
	// SddNode* sdd3 = sdd_read(sdd3_name.c_str(), sdd_manager);
	
	PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(sdd_manager_vtree(sdd_manager));
	PsddNode* psdd1 = psdd_manager->ConvertSddToPsdd(sdd1, sdd_manager_vtree(sdd_manager), 0); // use same vtree for both the psdds
	// PsddNode* psdd1 = psdd_manager->ReadPsddFile(psdd1_name.c_str(), 0);
	PsddNode* psdd2 = psdd_manager->ConvertSddToPsdd(sdd2, sdd_manager_vtree(sdd_manager), 0);
	// PsddNode* psdd3 = psdd_manager->ConvertSddToPsdd(sdd3, sdd_manager_vtree(sdd_manager), 0);

	auto result_pair = psdd_manager->Multiply(psdd1, psdd2, 0);
	PsddParameter param = result_pair.second;
	cout << result_pair.first << " " << param.parameter() << endl;
    const char filename[] = "5x5_icaps_l5_mult.psdd";
    psdd_node_util::WritePsddToFile(result_pair.first, filename);

	// auto result_pair1 = psdd_manager->Multiply(result_pair.first, psdd3, 0);	
	// PsddParameter param = result_pair1.second;
	// cout << result_pair1.first << " " << param.parameter() << endl;
	// const char filename[] = "coverage_mult.psdd";
	// psdd_node_util::WritePsddToFile(result_pair1.first, filename);

	// DIR *dir;
	// struct dirent *ent;
	// string dir_sdd = "/home/kushagra06/psdd/grid_sdds/";
	// string dir_psdd = "/home/kushagra06/psdd/grid_psdds/";
	// vector<string> sdd_files;
	// unordered_map<string, Vtree*> vtrees;
	// if ( (dir = opendir (dir_sdd.c_str())) != NULL) 
	// {
 //  /* print all the files and directories within directory */
	// 	while ((ent = readdir (dir)) != NULL) 
	// 	{
	// 		string file_name(ent->d_name);
	// 		int l = file_name.length();
	// 		char last_char = file_name[l-1];
	// 		if(last_char == 'e') // is a vtree
	// 		{
	// 			string s = dir_sdd + file_name;
	// 			const char* vtree_file = s.c_str();
	// 			Vtree* vtree = sdd_vtree_read(vtree_file);
	// 			size_t pos = file_name.find(".");
	// 			string vtree_name = file_name.substr(0, pos);
	// 			vtrees[vtree_name] = vtree;
	// 		}
	// 		else if(last_char == 'd') sdd_files.push_back(file_name);
 //    // cout << ent->d_name << cout;
	// 	}
	// 	closedir (dir);
	// } 
	// else
 // 	{
	//   /* could not open directory */
	//   perror ("");
	//   return EXIT_FAILURE;
	// }

	// vector<string> psdd_files;
	// if ( (dir = opendir (dir_psdd.c_str())) != NULL)
	// {
	// 	while ((ent = readdir(dir)) != NULL) 
	// 		if (ent->d_name[0]=='c') psdd_files.push_back(ent->d_name);
	// }
	
	// // for(auto i: vtrees) cout << i.first << " " << i.second << endl;

	// unordered_map<string, SddNode*> sdds;
	// Vtree* sdd_vtree;
	// /* Reading and storing sdds */
	// for(int i=0; i<sdd_files.size(); ++i)
	// {
	// 	string cluster_name;
	// 	if(sdd_files[i][0] != 'r') cluster_name = sdd_files[i][1];
	// 	else cluster_name = "root";
	// 	sdd_vtree = vtrees[cluster_name];
	// 	string sdd_file = dir_sdd + sdd_files[i];
	// 	SddManager* sdd_manager = sdd_manager_new(sdd_vtree);
	// 	SddNode* cur_sdd = sdd_read(sdd_file.c_str(), sdd_manager);
	// 	size_t pos = sdd_files[i].find(".");
	// 	string sdd_name = sdd_files[i].substr(0, pos);
	// 	sdds[sdd_name] = cur_sdd;
	// }
	
	// // unordered_map<string, PsddNode*> psdds;
	// // for(auto i: sdds)
	// // {
	// // 	string sdd_name = i.first;
	// // 	string cluster_name(sdd_name);
	// // 	if(cluster_name != "root") cluster_name = sdd_name[1];
	// // 	PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(vtrees[cluster_name]);
	// // 	PsddNode* psdd_node = psdd_manager->ConvertSddToPsdd(sdds[sdd_name], vtrees[cluster_name], 0);
	// // 	psdds[sdd_name] = psdd_node;
	// // }
	
	// // for(auto i: psdds) cout << i.first << " " << i.second << endl;
	
	// // PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(vtrees["1"]);
	// // auto result_pair = psdd_manager->Multiply(psdds[])
	

	// unordered_map<string, PsddNode*> psdds;
	// // Vtree* psdd_vtree = vtrees["1"];
	// // PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(psdd_vtree);
	// PsddNode* cur_psdd;
	// for(int i=0; i<psdd_files.size(); ++i) 
	// {
	// 	string cluster_name;
	// 	if(psdd_files[i][0] == 'r') cluster_name = "root";
	// 	else cluster_name.push_back(psdd_files[i][1]);
	// 	Vtree* psdd_vtree = vtrees[cluster_name];
	// 	string psdd_file = dir_psdd + psdd_files[i];
	// 	PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(psdd_vtree);
	// 	cur_psdd = psdd_manager->ReadPsddFile(psdd_file.c_str(), 0);
	// 	size_t pos = psdd_files[i].find(".");
	// 	string psdd_name = psdd_files[i].substr(0, pos);
	// 	psdds[psdd_name] = cur_psdd;
	// }
	// // // for(auto i: vtrees) cout << typeid(i.first).name() << " " << typeid(i.second).name() << endl;
	// PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(vtrees["2"]); 
	// auto result_pair = psdd_manager->Multiply(psdds["c2"], psdds["c5"], 0);
	// PsddParameter param = result_pair.second;
	// cout << param.parameter() << endl;
	// cout << result_pair.first << endl;

	// for(int i=2; i<psdds.size(); i++) res = psdd_manager->Multiply(res.first, psdds[i], 0);
	// cout << res.first << endl;
	// PsddManager *psdd_manager = PsddManager::GetPsddManagerFromVtree(psdd_vtree);
	// PsddNode *psdd = psdd_manager->ReadPsddFile(psdd_filename, 0);

	
return 0;

}
