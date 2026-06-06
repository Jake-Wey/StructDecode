#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>

namespace struct_decode {

using DFAState = int;

constexpr DFAState DEAD_STATE = -1;
constexpr DFAState START_STATE = 0;

using TransitionMap = std::unordered_map<char, DFAState>;

/// <summary>
/// A DFA for matching regular repressions.
/// </summary>
class DFA {
public:
	DFA();

	/// <summary>
	/// Get the current state.
	/// </summary>
	/// <returns>Current DFA state.</returns>
	DFAState current_state() const { return current_state_; }

	/// <summary>
	/// Reset to the start state.
	/// </summary>
	void reset();

	/// <summary>
	/// Advance the DFA by one character.
	/// </summary>
	/// <param name="c">The input character.</param>
	/// <returns>The new state.</returns>
	DFAState advance(char c);

	/// <summary>
	/// Check if current state is accepting.
	/// </summary>
	/// <returns>true if the current state is an accepting state.</returns>
	bool is_accepting() const;

	/// <summary>
	/// Check if current state is dead (no valid transitions).
	/// </summary>
	/// <returns>true if the DFA is in a dead state.</returns>
	bool is_dead() const;

	/// <summary>
	/// Get the set of accepting states.
	/// </summary>
	/// <returns>Set of accepting state IDs.</returns>
	const std::unordered_set<DFAState>& accepting_states() const { return accepting_states_; }

	/// <summary>
	/// Get valid next characters from current state.
	/// </summary>
	/// <returns>Set of characters that have valid transitions.</returns>
	std::unordered_set<char> get_valid_next_chars() const;

	/// <summary>
	/// Get valid next characters from a specific state.
	/// </summary>
	/// <param name="state">The state to check.</param>
	/// <returns>Set of characters that have valid transitions.</returns>
	std::unordered_set<char> get_valid_next_chars(DFAState state) const;

	/// <summary>
	/// Get the number of states.
	/// </summary>
	/// <returns>Number of states in the DFA.</returns>
	size_t num_states() const { return transitions_.size(); }

protected:
	DFAState current_state_;
	std::vector<TransitionMap> transitions_;
	std::unordered_set<DFAState> accepting_states_;

	void add_state();
	void add_transition(DFAState from, char c, DFAState to);
	void set_accepting(DFAState state);
};

/// <summary>
/// DFA for matching a literal string.
/// </summary>
class LiteralDFA : public DFA {
public:
	explicit LiteralDFA(const std::string& literal);
};

/// <summary>
/// DFA for matching character classes (e.g. [a-z])
/// </summary>
class CharClassDFA : public DFA {
public:
	/// <summary>
	/// Create a DFA for a character class.
	/// </summary>
	/// <param name="chars">String containing all characters in the class.</param>
	/// <param name="negated">If true, matches any character NOT in the class.</param>
	CharClassDFA(const std::string& chars, bool negated = false);
};

/// <summary>
/// DFA for matching patterns like [a-z]+.
/// </summary>
class RepeatingCharClassDFA : public DFA {
public:
	RepeatingCharClassDFA(
		const std::string& chars,
		int min_reps = 1,
		int max_reps = -1
	);
};

/// <summary>
/// DFA for matching a sequence of sub-DFAs.
/// </summary>
class SequenceDFA : public DFA {
public:
	explicit SequenceDFA(const std::vector<std::unique_ptr<DFA>>& dfas);
};

class AlternationDFA : public DFA {
public:
	explicit AlternationDFA(const std::vector<std::unique_ptr<DFA>>& dfas);
};

/// <summary>
/// A simplified regex parser and DFA builder.
/// </summary>
class RegexDFA {
public:
	/// <summary>
	/// Parse a regex pattern and build a DFA.
	/// </summary>
	/// <param name="pattern">The regex pattern.</param>
	/// <returns>A DFA for matching the pattern.</returns>
	static std::unique_ptr<DFA> parse(const std::string& pattern);

	/// <summary>
	/// Check if a pattern is supported.
	/// </summary>
	/// <param name="pattern">The regex pattern.</param>
	/// <returns>true if the pattern can be parsed.</returns>
	static bool is_supported(const std::string& pattern);
};

} // struct_decode