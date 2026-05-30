#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace struct_decode {

/// <summary>
/// Trie node for storing token IDs and their string representation.
/// Each node represents a character in the token string, with leaf nodes
/// containing the actual token ID.
/// </summary>
struct TrieNode{
	std::unordered_map<char, std::unique_ptr<TrieNode>> children;
	int token_id = -1; // -1 means not a complete token
	bool is_end = false;
};

/// <summary>
/// A trie for efficient token lookups.
/// </summary>
class TokenTrie {
public:
	TokenTrie();

	/// <summary>
	/// Insert a token into the trie.
	/// </summary>
	/// <param name="token_str">The string representation of the token.</param>
	/// <param name="token_id">The ID of the token.</param>
	void insert(const std::string& token_str, int token_id);

	/// <summary>
	/// Check if a string exists in the trie.
	/// </summary>
	/// <param name="str">The string to check.</param>
	/// <returns>true if the string exists as a complete token.</returns>
	bool contains(const std::string& str) const;

	/// <summary>
	/// Get the token ID for a string.
	/// </summary>
	/// <param name="str">The string to look up.</param>
	/// <returns>The token ID, or -1 if not found.</returns>
	int get_token_id(const std::string& str) const;

	/// <summary>
	/// Find all tokens that start with the given prefix.
	/// </summary>
	/// <param name="prefix">The prefix to search for.</param>
	/// <returns>Vector of (token_id, token_str) pairs.</returns>
	std::vector<std::pair<int, std::string>> find_by_prefix(const std::string& prefix) const;

	/// <summary>
	/// Get all token IDs in the trie.
	/// </summary>
	/// <returns>Vector of all token IDs.</returns>
	std::vector<int> get_all_token_ids() const;

	/// <summary>
	/// Get the root node of the trie.
	/// </summary>
	/// <returns>Pointer to the root node.</returns>
	const TrieNode* get_root() const { return root_.get(); }

	/// <summary>
	/// Get the number of tokens in the trie.
	/// </summary>
	/// <returns>Number of tokens.</returns>
	size_t size() const { return size_; }

private:
	std::unique_ptr<TrieNode> root_;
	size_t size_ = 0;

	void find_all_from_node(
		const TrieNode* node,
		const std::string& prefix,
		std::vector<std::pair<int, std::string>>& results
	) const;
};

} // struct_decode