#include "token_trie.h"

namespace struct_decode {

TokenTrie::TokenTrie() : root_(std::make_unique<TrieNode>()) {}

void TokenTrie::insert(const std::string& token_str, int token_id) {
	TrieNode* current = root_.get();

	for (char c : token_str) {
		if (current->children.find(c) == current->children.end()) {
			current->children[c] = std::make_unique<TrieNode>();
		}
		current = current->children[c].get();
	}

	current->token_id = token_id;
	current->is_end = true;
	size_++;
}

bool TokenTrie::contains(const std::string& str) const {
	const TrieNode* node = root_.get();

	for (char c : str) {
		auto it = node->children.find(c);
		if (it == node->children.end()) {
			return false;
		}
		node = it->second.get();
	}

	return node->is_end;
}

int TokenTrie::get_token_id(const std::string& str) const {
	const TrieNode* node = root_.get();
	
	for (char c : str) {
		auto it = node->children.find(c);
		if (it == node->children.end()) {
			return -1;
		}
		node = it->second.get();
	}

	return node->token_id;
}

std::vector<std::pair<int, std::string>> TokenTrie::find_by_prefix(const std::string& prefix) const {
	std::vector<std::pair<int, std::string>> results;

	const TrieNode* current = root_.get();
	for (char c : prefix) {
		auto it = current->children.find(c);
		if (it == current->children.end()) {
			return results;
		}
		current = it->second.get();
	}

	find_all_from_node(current, prefix, results);

	return results;
}

std::vector<int> TokenTrie::get_all_token_ids() const {
	std::vector<int> ids;
	std::vector<std::pair<int, std::string>> results;
	find_all_from_node(root_.get(), "", results);
	ids.reserve(results.size());
	for (const auto& [id, str] : results) {
		ids.push_back(id);
	}
	return ids;
}

void TokenTrie::find_all_from_node(
	const TrieNode* node, 
	const std::string& prefix, 
	std::vector<std::pair<int, std::string>>& results
) const {
	if (node->is_end) {
		results.emplace_back(node->token_id, prefix);
	}

	for (const auto& [c, child] : node->children) {
		find_all_from_node(child.get(), prefix + c, results);
	}
}

} // struct_decode